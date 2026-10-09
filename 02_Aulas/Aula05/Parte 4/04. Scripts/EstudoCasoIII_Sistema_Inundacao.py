# -*- coding: utf-8 -*-
"""HydroUAI | Piracicaba: cinco LSTMs diretas (1 a 5 horas).

Origem t, dados agregados para escala horaria:
  - Chuva PLU(mm)713: soma dos seis valores de 10 min de cada hora.
  - Niveis FLU(m)713 e FLU(m)46: media dos seis valores de cada hora.
  - PLU(mm)713 e FLU(m)713: 6 horas de entrada, t-8h ... t-3h,
    representando a janela de 6 horas centrada aproximadamente em t-6h.
  - FLU(m)46: 6 horas de entrada, t-5h ... t.
  - Alvo: FLU(m)46 em t+1h, t+2h, t+3h, t+4h e t+5h.
  - Cada registro horario e rotulado pelo FINAL da hora (closed/label right).
    Exigem-se seis observacoes validas de 10 min por hora; horas incompletas
    ficam NaN, sem transformar chuva faltante em zero.

Cada horizonte treina uma LSTM independente, com as mesmas tres entradas.
Divisao cronologica: 80% treinamento e 20% validacao (sem teste).
Para uso no Spyder, coloque este arquivo em '02. Scripts' e o TXT em
'01. Arquivo de Texto', ambos sob a mesma pasta principal. Tambem aceita
um TXT na mesma pasta do script ou um caminho configurado em DADOS.
"""

# %% 1 | Bibliotecas e configuracoes
from pathlib import Path
import copy
import json
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

PASTA = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
DADOS = PASTA.parent / '01. Arquivo de Texto' / 'dados_interpolados.txt'
if not DADOS.exists():
    DADOS = PASTA / 'dados_interpolados.txt'
SAIDA = PASTA / 'resultados'

ALVO = 'FLU(m)46'
ENTRADAS = ['PLU(mm)713', 'FLU(m)713', ALVO]
PASSOS_JANELA = 6
ATRASO_713 = 3              # Ultimo registro da 713: t-3h
ATRASO_EXUTORIO = 0         # Ultimo valor do exutorio: t
HORIZONTES = [1, 2, 3, 4, 5]  # 1h, 2h, 3h, 4h, 5h

GRID_CAMADAS = [1, 2]
GRID_NEURONIOS = [64, 128]
# Quatro combinacoes por horizonte; selecao pelo menor MSE de validacao.
EPOCAS = 60
BATCH_SIZE = 512
TAXA_APRENDIZADO = 5e-4
PACIENCIA = 12
PACIENCIA_LR = 3
PASSO_TREINO = 1            # Cada origem ja corresponde a uma hora
SEMENTE = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% 2 | Funcoes de leitura e verificacao do calendario

def ler_dados(caminho):
    if not caminho.exists():
        raise FileNotFoundError(f'Arquivo nao encontrado: {caminho}')
    df = pd.read_csv(caminho, sep='\t', usecols=['DATA'] + ENTRADAS[:2] + [ALVO])
    df['DATA'] = pd.to_datetime(df['DATA'], format='%m/%d/%Y %H:%M')
    df = df.set_index('DATA').sort_index()
    if df.index.has_duplicates or df.index.hasnans:
        raise ValueError('Datas duplicadas ou invalidas.')
    if not (df.index.to_series().diff().iloc[1:] == pd.Timedelta(minutes=10)).all():
        raise ValueError('Os dados precisam ter intervalo continuo de 10 minutos.')
    df = df.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    # Resample horario: cada hora termina no rotulo (ex.: 13:00 inclui 12:10..13:00).
    # min_count=6 impede que intervalos com falhas virem chuvas artificiais.
    chuva = df[['PLU(mm)713']].resample('1h', closed='right', label='right').sum(min_count=6)
    niveis = df[['FLU(m)713', ALVO]].resample('1h', closed='right', label='right').agg(['mean', 'count'])
    horario = chuva.copy()
    for coluna in ['FLU(m)713', ALVO]:
        horario[coluna] = niveis[(coluna, 'mean')].where(niveis[(coluna, 'count')] == 6)
    # A primeira/ultima hora podem ser incompletas; mantemos NaN para auditoria.
    return horario[ENTRADAS]

# %% 3 | Classe de janelas e divisao temporal

class BaseJanelas:
    """Monta lotes sob demanda para evitar materializar todas as janelas na RAM."""

    def __init__(self, df, horizonte):
        self.df = df
        self.horizonte = horizonte
        self.n = len(df)
        self.c1 = int(.80 * self.n)
        self.raw = df[ENTRADAS].to_numpy(dtype=np.float32)
        self.q = df[ALVO].to_numpy(dtype=np.float32)
        # Janela 713: t-8,...,t-3; exutorio: t-5,...,t (horas).
        self.deslocamentos = np.stack([
            np.arange(ATRASO_713 + PASSOS_JANELA - 1, ATRASO_713 - 1, -1),
            np.arange(ATRASO_713 + PASSOS_JANELA - 1, ATRASO_713 - 1, -1),
            np.arange(PASSOS_JANELA - 1, -1, -1),
        ], axis=1)  # (6, 3), atrasos em horas
        origens = np.arange(ATRASO_713 + PASSOS_JANELA - 1,
                            self.n - horizonte, dtype=np.int64)
        valido = np.isfinite(self.q[origens + horizonte])
        for j in range(3):
            faltas = np.r_[0, np.cumsum(~np.isfinite(self.raw[:, j]))]
            atraso_min = int(self.deslocamentos[-1, j])
            atraso_max = int(self.deslocamentos[0, j])
            inicio = origens - atraso_max
            fim = origens - atraso_min + 1
            valido &= (faltas[fim] - faltas[inicio]) == 0
        # Origem e alvo devem estar no mesmo bloco cronologico.
        blocos = np.zeros(self.n, dtype=np.int8)
        blocos[self.c1:] = 1
        valido &= blocos[origens] == blocos[origens + horizonte]
        origens = origens[valido]
        self.indices = {nome: origens[blocos[origens] == k]
                        for k, nome in enumerate(['treino', 'validacao'])}
        self.indices['treino'] = self.indices['treino'][::PASSO_TREINO]
        if any(len(ids) < 10 for ids in self.indices.values()):
            raise ValueError(f'Poucos exemplos validos para h={horizonte}.')
        # Ajuste de normalizacao APENAS com dados do treino, sem vazamento.
        self.sx = StandardScaler().fit(self.raw[:self.c1])
        self.sy = StandardScaler().fit(self.q[self.indices['treino'] + horizonte, None])
        self.x = self.sx.transform(self.raw).astype(np.float32)
        self.y = self.sy.transform(self.q[:, None]).astype(np.float32)

    def lote(self, origens):
        pos = origens[:, None, None] - self.deslocamentos[None, :, :]
        # Uma sequencia temporal de 6 horas x 3 atributos.
        x = np.stack([self.x[pos[:, :, j], j] for j in range(3)], axis=2)
        y = self.y[origens + self.horizonte]
        return (torch.from_numpy(x).to(DEVICE),
                torch.from_numpy(y).to(DEVICE))

# %% 4 | Modelo LSTM e rotina de treinamento

class LSTMPrevisao(nn.Module):
    def __init__(self, camadas, neuronios):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=neuronios,
                            num_layers=camadas, batch_first=True)
        self.saida = nn.Linear(neuronios, 1)

    def forward(self, x):
        sequencia, _ = self.lstm(x)
        return self.saida(sequencia[:, -1, :])


def fixar_semente(semente):
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semente)


@torch.no_grad()
def prever_normalizado(modelo, base, ids):
    modelo.eval()
    partes = []
    for i in range(0, len(ids), BATCH_SIZE):
        x, _ = base.lote(ids[i:i+BATCH_SIZE])
        partes.append(modelo(x).cpu().numpy())
    return np.concatenate(partes, axis=0)


def treinar(base, camadas, neuronios):
    fixar_semente(SEMENTE)
    modelo = LSTMPrevisao(camadas, neuronios).to(DEVICE)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=TAXA_APRENDIZADO)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        otimizador, mode='min', factor=.5, patience=PACIENCIA_LR, min_lr=1e-6)
    melhor_mse = float('inf')
    melhor_estado = None
    melhor_epoca = 0
    espera = 0
    historico = []
    for epoca in range(1, EPOCAS + 1):
        modelo.train()
        ordem = np.random.permutation(base.indices['treino'])
        soma = 0.0
        lr = otimizador.param_groups[0]['lr']
        for inicio in range(0, len(ordem), BATCH_SIZE):
            ids = ordem[inicio:inicio+BATCH_SIZE]
            x, y = base.lote(ids)
            otimizador.zero_grad(set_to_none=True)
            perda = nn.functional.mse_loss(modelo(x), y)
            if not torch.isfinite(perda):
                raise ValueError('Perda nao finita: verifique os dados.')
            perda.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 5.0)
            otimizador.step()
            soma += perda.item() * len(ids)
        ids_val = base.indices['validacao']
        prev_val = prever_normalizado(modelo, base, ids_val)
        mse_val = float(np.mean((prev_val - base.y[ids_val + base.horizonte]) ** 2))
        historico.append({'epoca': epoca, 'mse_treino': soma/len(ordem),
                          'mse_validacao': mse_val, 'lr': lr})
        scheduler.step(mse_val)
        if mse_val < melhor_mse:
            melhor_mse, melhor_epoca, espera = mse_val, epoca, 0
            melhor_estado = copy.deepcopy(modelo.state_dict())
        else:
            espera += 1
        if epoca == 1 or epoca % 5 == 0:
            print(f'  Epoca {epoca:02d} | MSE treino={soma/len(ordem):.5f}'
                  f' | validacao={mse_val:.5f}', flush=True)
        if espera >= PACIENCIA:
            print('  Early stopping.')
            break
    modelo.load_state_dict(melhor_estado)
    return modelo, pd.DataFrame(historico), melhor_epoca, melhor_mse

def metricas(observado, previsto):
    o, p = np.asarray(observado).ravel(), np.asarray(previsto).ravel()
    erro = p - o
    den = np.sum((o-o.mean())**2)
    nse = 1 - np.sum(erro**2)/den if den > 0 else np.nan
    kge = np.nan
    if o.std() > 0 and p.std() > 0 and abs(o.mean()) > 1e-12:
        r = np.corrcoef(o, p)[0, 1]
        kge = 1 - np.sqrt((r-1)**2 + (p.std()/o.std()-1)**2
                          + (p.mean()/o.mean()-1)**2)
    pbias = 100 * np.sum(p-o)/np.sum(o) if abs(np.sum(o)) > 1e-12 else np.nan
    return {'RMSE_m': float(np.sqrt(np.mean(erro**2))),
            'MAE_m': float(np.mean(abs(erro))),
            'NSE': float(nse), 'KGE': float(kge), 'PBIAS_pct': float(pbias)}



# %% 5 | Ler os dados (execute esta celula primeiro)
# No Spyder: Ctrl+Enter executa somente a celula atual.
# Configure Graficos > Backend = Inline para ver os plots no painel Plots.
plt.rcParams.update({'figure.figsize': (12, 4.5), 'axes.grid': True,
                     'grid.alpha': .2})
df = ler_dados(DADOS)
print('Periodo horario:', df.index.min(), 'ate', df.index.max())
print('Resolucao: 1 hora | chuva somada; niveis medios')
print('Registros horarios:', len(df), '| faltantes por coluna:')
print(df.isna().sum())

# %% 6 | Serie completa e divisao temporal 80% / 20%
corte = int(.8 * len(df))
data_corte = df.index[corte]
fig, ax = plt.subplots(figsize=(14, 4.5))
ax.plot(df.index, df[ALVO], color='#243746', lw=.8, label='Nivel observado no exutorio')
# Hachuras com cores transparentes, sem encobrir a serie.
ax.axvspan(df.index[0], data_corte, facecolor='#2563eb', alpha=.08,
           hatch='///', edgecolor='#2563eb', linewidth=0, label='Treinamento (80%)')
ax.axvspan(data_corte, df.index[-1], facecolor='#e67e22', alpha=.10,
           hatch='xxx', edgecolor='#e67e22', linewidth=0, label='Validacao (20%)')
ax.axvline(data_corte, color='#333333', linestyle='--', lw=1)
ax.set(title='Serie completa e divisao cronologica', ylabel='Nivel no exutorio (m)',
       xlabel='Data')
ax.legend(loc='best')
plt.tight_layout()
plt.show()
print('Inicio da validacao:', data_corte)
print('Primeiras horas agregadas:')
print(df.head(8))

# %% 7 | Visualizar a logica das janelas de entrada
# Selecione uma origem ilustrativa, com historico suficiente.
ORIGEM_EXEMPLO = 100
instante = df.index[ORIGEM_EXEMPLO]
fig, axs = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
for ax, coluna, atraso in zip(axs, ENTRADAS, [ATRASO_713, ATRASO_713, ATRASO_EXUTORIO]):
    i0 = ORIGEM_EXEMPLO - atraso - PASSOS_JANELA + 1
    i1 = ORIGEM_EXEMPLO - atraso + 1
    ax.plot(df.index[i0:i1], df[coluna].iloc[i0:i1], 'o-', ms=2.5)
    ax.axvline(instante, color='black', linestyle='--', label='Origem t')
    ax.set(ylabel=coluna)
    ax.legend(loc='upper left')
axs[0].set_title('Entradas: 713 centrada em t-6h; exutorio ate t')
axs[-1].set_xlabel('Data e hora')
plt.tight_layout()
plt.show()

# %% 8 | Preparar as cinco bases de janelas
bases = {h: BaseJanelas(df, h) for h in HORIZONTES}
for h, base in bases.items():
    print(f'+{h}h:', {nome: len(ids) for nome, ids in base.indices.items()})
# Inspecionar o formato de uma sequencia: (lote, 6 horas, 3 entradas)
x_exemplo, y_exemplo = bases[1].lote(bases[1].indices['treino'][:4])
print('X:', tuple(x_exemplo.shape), '| y:', tuple(y_exemplo.shape))

# %% 9 | Grid search didatico: horizonte de 1 hora
# Treina quatro arquiteturas e escolhe a de menor MSE de validacao.
# Cada configuracao parte da mesma semente e tem seu proprio early stopping.
import itertools

modelos = {}
historicos = {}
melhores_epocas = {}
melhores_configs = {}
resultados_grid = []


def buscar_melhor_modelo(h):
    base = bases[h]
    melhor = None
    for camadas, neuronios in itertools.product(GRID_CAMADAS, GRID_NEURONIOS):
        print(f'\n+{h}h | camadas={camadas}, neuronios={neuronios}', flush=True)
        modelo, hist, epoca, mse = treinar(base, camadas, neuronios)
        resultados_grid.append({'horizonte_h': h, 'camadas': camadas,
                                'neuronios': neuronios, 'melhor_epoca': epoca,
                                'mse_validacao': mse,
                                'rmse_validacao_m': np.sqrt(mse) * base.sy.scale_[0]})
        if melhor is None or mse < melhor['mse']:
            melhor = {'modelo': modelo, 'historico': hist,
                      'epoca': epoca, 'mse': mse,
                      'config': {'camadas': camadas, 'neuronios': neuronios}}
    modelos[h] = melhor['modelo']
    historicos[h] = melhor['historico']
    melhores_epocas[h] = melhor['epoca']
    melhores_configs[h] = melhor['config']
    print(f"\nMelhor +{h}h: {melhor['config']} | "
          f"RMSE validacao = {np.sqrt(melhor['mse']) * base.sy.scale_[0]:.4f} m")


buscar_melhor_modelo(1)
print(pd.DataFrame(resultados_grid).to_string(index=False))

# %% 10 | Curva de aprendizado da melhor arquitetura para +1h
hist = historicos[1]
escala = bases[1].sy.scale_[0]
fig, ax = plt.subplots()
ax.plot(hist.epoca, np.sqrt(hist.mse_treino) * escala, label='Treino')
ax.plot(hist.epoca, np.sqrt(hist.mse_validacao) * escala, label='Validacao')
ax.set(xlabel='Epoca', ylabel='RMSE (m)',
       title=f"Aprendizado | +1h | {melhores_configs[1]}")
ax.legend()
plt.tight_layout()
plt.show()

# %% 11 | Grid search para os outros quatro horizontes
# Ao final: 5 horizontes x 4 configuracoes = 20 treinamentos.
for H in [2, 3, 4, 5]:
    buscar_melhor_modelo(H)

ranking_grid = pd.DataFrame(resultados_grid).sort_values(
    ['horizonte_h', 'mse_validacao'])
print('\nRanking por horizonte (menor MSE de validacao primeiro):')
print(ranking_grid.to_string(index=False))

# %% 12 | Curvas de aprendizado dos cinco modelos selecionados
fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
for h in HORIZONTES:
    hist = historicos[h]
    escala = bases[h].sy.scale_[0]
    axs[0].plot(hist.epoca, np.sqrt(hist.mse_treino) * escala,
                label=f'+{h}h')
    axs[1].plot(hist.epoca, np.sqrt(hist.mse_validacao) * escala,
                label=f'+{h}h')
axs[0].set(title='Treinamento', xlabel='Epoca', ylabel='RMSE (m)')
axs[1].set(title='Validacao', xlabel='Epoca', ylabel='RMSE (m)')
for ax in axs:
    ax.legend()
plt.tight_layout()
plt.show()

# %% 13 | Calcular previsoes e metricas (sem salvar graficos)
previsoes = {}
linhas_metricas = []
for h in HORIZONTES:
    base = bases[h]
    previsoes[h] = {}
    for bloco, ids in base.indices.items():
        estimado = base.sy.inverse_transform(
            prever_normalizado(modelos[h], base, ids)).ravel()
        observado = base.q[ids + h]
        tabela = pd.DataFrame({'origem': df.index[ids],
                               'data_alvo': df.index[ids + h],
                               'observado': observado, 'previsto': estimado})
        previsoes[h][bloco] = tabela
        linhas_metricas.append({'horizonte_h': h, 'bloco': bloco,
                                'n': len(tabela), **metricas(observado, estimado)})
metricas_df = pd.DataFrame(linhas_metricas)
print(metricas_df.to_string(index=False))

# %% 14 | Comparar desempenho na validacao
val = metricas_df.query("bloco == 'validacao'").sort_values('horizonte_h')
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(val.horizonte_h, val.NSE, 'o-')
axs[1].plot(val.horizonte_h, val.KGE, 'o-')
axs[0].set(title='NSE | validacao', ylabel='NSE', xlabel='Horizonte (h)')
axs[1].set(title='KGE | validacao', ylabel='KGE', xlabel='Horizonte (h)')
for ax in axs: ax.set_xticks([1, 2, 3, 4, 5])
plt.tight_layout()
plt.show()

# %% 15 | Hidrograma observado e previsto na validacao (escolha o horizonte)
H_VISUALIZAR = 1  # 1=1h; 2=2h; 3=3h; 4=4h; 5=5h
tabela = previsoes[H_VISUALIZAR]['validacao']
fig, ax = plt.subplots(figsize=(14, 4.5))
ax.plot(tabela.data_alvo, tabela.observado, label='Observado', lw=1)
ax.plot(tabela.data_alvo, tabela.previsto, label='LSTM', lw=1, alpha=.8)
ax.set(title=f'Validacao | horizonte +{H_VISUALIZAR//6}h',
       ylabel='Nivel do exutorio (m)', xlabel='Data')
ax.legend()
plt.tight_layout()
plt.show()

# %% 16 | Escolher a data de origem da previsao de 5 horas
# Use None para selecionar automaticamente uma origem valida na validacao.
# Ou informe, por exemplo: DATA_ESCOLHIDA = '2020-01-15 12:00'
DATA_ESCOLHIDA = None  # Ou '2022-01-30 15:00', se a data estiver na validacao
# A origem precisa ter previsoes validas para TODOS os cinco horizontes.
origens_comuns = set(previsoes[1]['validacao']['origem'])
for h in HORIZONTES[1:]:
    origens_comuns &= set(previsoes[h]['validacao']['origem'])
origens_comuns = pd.DatetimeIndex(sorted(origens_comuns))
# Exigir 5h observadas antes e depois da origem.
origens_comuns = origens_comuns[(origens_comuns >= df.index[5]) &
                                (origens_comuns <= df.index[-6])]
if len(origens_comuns) == 0:
    raise ValueError('Nenhuma origem comum valida com 5h antes/depois.')
if DATA_ESCOLHIDA is None:
    # Origem de maior nivel observado, para um exemplo visualmente interessante.
    origem = df.loc[origens_comuns, ALVO].idxmax()
else:
    origem = pd.Timestamp(DATA_ESCOLHIDA)
    if origem not in origens_comuns:
        raise ValueError('Data indisponivel. Escolha uma data presente em origens_comuns.')
print('Origem escolhida:', origem)

# %% 17 | Plot final: observado de t-5h a t+5h e 5 previsoes diretas
# Observado: 5 horas antes e 5 horas depois da origem.
# Previsoes: cinco pontos independentes +1h, +2h, +3h, +4h, +5h.
janela_obs = df.loc[origem-pd.Timedelta(hours=5):
                    origem+pd.Timedelta(hours=5), ALVO]
valores_previstos = []
for h in HORIZONTES:
    tabela = previsoes[h]['validacao']
    linha = tabela.loc[tabela['origem'] == origem]
    if len(linha) != 1:
        raise ValueError(f'Origem sem previsao unica para +{h}h.')
    valores_previstos.append(float(linha['previsto'].iloc[0]))
datas_futuras = [origem + pd.Timedelta(hours=h) for h in HORIZONTES]
q_origem = float(df.loc[origem, ALVO])
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(janela_obs.index, janela_obs.values, color='#0072B2', lw=1.8,
        label='Nivel observado (t-5h ate t+5h)')
ax.plot([origem] + datas_futuras, [q_origem] + valores_previstos,
        'o-', color='#D55E00', lw=1.8, ms=6,
        label='Cinco LSTMs diretas (1 a 5h)')
ax.axvline(origem, color='black', ls='--', lw=1, label='Origem da previsao')
ax.axvspan(origem, origem + pd.Timedelta(hours=5), color='#D55E00', alpha=.06)
ax.set(title=f'Previsao de 5 horas | origem {origem:%d/%m/%Y %H:%M}',
       xlabel='Data e hora', ylabel='Nivel no exutorio (m)')
ax.legend(loc='best')
plt.tight_layout()
plt.show()

# %% 18 | Opcional: salvar modelos, escalonadores, tabelas e metricas
# Nada e salvo automaticamente. Execute esta celula somente se desejar.
SAIDA.mkdir(parents=True, exist_ok=True)
metricas_df.to_csv(SAIDA / 'metricas.csv', index=False)
pd.DataFrame(resultados_grid).to_csv(SAIDA / 'grid_search.csv', index=False)
for h in HORIZONTES:
    pasta_h = SAIDA / f'horizonte_{h}h'
    pasta_h.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': modelos[h].state_dict(), 'horizonte_horas': h,
                'entradas': ENTRADAS, 'passos_janela': PASSOS_JANELA,
                'deslocamentos_horas': bases[h].deslocamentos.tolist(),
                'resolucao': '1h', 'chuva': 'soma', 'niveis': 'media',
                'camadas': melhores_configs[h]['camadas'],
                 'neuronios': melhores_configs[h]['neuronios']},
               pasta_h / 'modelo.pt')
    with open(pasta_h / 'scalers.pkl', 'wb') as f:
        pickle.dump({'x': bases[h].sx, 'y': bases[h].sy}, f)
    historicos[h].to_csv(pasta_h / 'aprendizado.csv', index=False)
    for bloco, tabela in previsoes[h].items():
        tabela.to_csv(pasta_h / f'previsoes_{bloco}.csv', index=False)
print('Arquivos salvos em:', SAIDA)

# %% 19 | Hidrograma de 17 horas + HEC-RAS Controller

import numpy as np
import pandas as pd
import shutil
import webbrowser
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
from branca.element import Element
from pathlib import Path

# --------------------------------------------------
# 1. CAMINHOS E CONFIGURAÇÕES
# --------------------------------------------------

PASTA_RAS = PASTA.parent / "03. HECRAS"

ARQUIVO_U01 = PASTA_RAS / "Case_Study.u01"
ARQUIVO_PRJ = PASTA_RAS / "Case_Study.prj"
CAMINHO_EDIFICIOS = PASTA.parent / "02. Shapefiles" / "Edificios.shp"
CAMINHO_MANCHA = PASTA_RAS / "Previsao_LSTM" / "Inundation Boundary (Max Value_0).shp"
CAMINHO_DEPTH = PASTA_RAS / "Previsao_LSTM" / "Depth (Max).MDT.MDT_v05"
PASTA_MAPAS = PASTA / "resultados_impactos"
PASTA_MAPAS.mkdir(parents=True, exist_ok=True)

INICIO_HIDRO = pd.Timestamp("2022-01-30 05:00")
ORIGEM_RAS = pd.Timestamp("2022-01-30 16:00")

A_CURVA = 73.642
H0_CURVA = 0.6768
B_CURVA = 1.3129

EXECUTAR_HECRAS = True
MOSTRAR_HECRAS = True
FECHAR_HECRAS_AO_FINAL = False  # False permite inspecionar os resultados
CONTROLLER_PROGID = "RAS701.HECRASCONTROLLER"

# --------------------------------------------------
# 2. CONVERTER NÍVEL EM VAZÃO
# --------------------------------------------------

def nivel_para_vazao(h):

    h = np.asarray(h, dtype=float)

    if np.any(~np.isfinite(h)) or np.any(h <= H0_CURVA):
        raise ValueError("Níveis inválidos para a curva-chave.")

    return A_CURVA * (h - H0_CURVA) ** B_CURVA


# --------------------------------------------------
# 3. RECUPERAR 12 NÍVEIS OBSERVADOS
# --------------------------------------------------

horas_observadas = pd.date_range(
    INICIO_HIDRO,
    ORIGEM_RAS,
    freq="h"
)

if not horas_observadas.isin(df.index).all():
    raise ValueError("Existem horários observados ausentes.")

niveis_obs = df.loc[horas_observadas, ALVO]

if niveis_obs.isna().any():
    raise ValueError("Existem níveis observados ausentes.")


# --------------------------------------------------
# 4. RECUPERAR 5 PREVISÕES DA LSTM
# --------------------------------------------------

valores_futuros = []

for h in range(1, 6):

    tabela = previsoes[h]["validacao"]

    linha = tabela.loc[
        tabela["origem"] == ORIGEM_RAS
    ]

    if len(linha) != 1:
        raise ValueError(f"Previsão +{h}h indisponível.")

    valores_futuros.append(
        float(linha["previsto"].iloc[0])
    )

horas_futuras = pd.date_range(
    ORIGEM_RAS + pd.Timedelta(hours=1),
    periods=5,
    freq="h"
)

# --------------------------------------------------
# 5. CONSTRUIR HIDROGRAMA DE 17 VALORES
# --------------------------------------------------

niveis_17 = pd.concat([
    niveis_obs,
    pd.Series(valores_futuros, index=horas_futuras)
])

if len(niveis_17) != 17:
    raise ValueError("O hidrograma precisa ter 17 valores.")

if not niveis_17.index.equals(
    pd.date_range(INICIO_HIDRO, periods=17, freq="h")
):
    raise ValueError("Os 17 horários não são consecutivos.")

hidrograma_17 = pd.DataFrame({
    "Nivel_m": niveis_17,
    "Vazao_m3s": nivel_para_vazao(niveis_17)
})

hidrograma_17["Origem"] = (
    ["Observado"] * 12 + ["Previsto"] * 5
)

print("\nHIDROGRAMA DE ENTRADA DO HEC-RAS")
print(hidrograma_17)


# --------------------------------------------------
# 6. EDITAR SOMENTE OS 17 VALORES DO .U01
# --------------------------------------------------

def atualizar_u01(caminho, vazoes, bc="BC_Montante"):

    caminho = Path(caminho)
    vazoes = np.asarray(vazoes, dtype=float)

    if not caminho.is_file():
        raise FileNotFoundError(caminho)

    if len(vazoes) != 17 or not np.isfinite(vazoes).all():
        raise ValueError("São necessárias 17 vazões válidas.")

    # Campos de largura fixa: exatamente 8 caracteres
    campos = [f"{q:8.2f}" for q in vazoes]

    if any(len(c) != 8 for c in campos):
        raise ValueError(
            "Uma vazão excede a largura de 8 caracteres."
        )

    # Preservar codificação e quebras de linha originais
    with open(caminho, "r", encoding="latin-1", newline="") as f:
        linhas = f.readlines()

    # Verificar integridade geral
    versoes = [
        i for i, linha in enumerate(linhas)
        if linha.startswith("Program Version=")
    ]

    if len(versoes) != 1:
        raise ValueError(
            "O arquivo .u01 contém estrutura duplicada "
            "ou inválida. Restaure o original antes de continuar."
        )

    # Localizar condições de contorno pelo último identificador
    locais = []

    for i, linha in enumerate(linhas):

        if linha.startswith("Boundary Location="):

            campos_bc = linha.split("=", 1)[1].split(",")

            if len(campos_bc) < 8:
                raise ValueError(
                    "Boundary Location com formato inválido."
                )

            nome_bc = campos_bc[7].strip()
            locais.append((i, nome_bc))

    montantes = [
        i for i, nome in locais if nome == bc
    ]

    jusantes = [
        i for i, nome in locais if nome == "BC_Jusante"
    ]

    if len(montantes) != 1 or len(jusantes) != 1:
        raise ValueError(
            "BC_Montante ou BC_Jusante ausente/duplicada. "
            "Edição cancelada."
        )

    inicio = montantes[0]

    proximos = [
        i for i, _ in locais if i > inicio
    ]

    fim = min(proximos) if proximos else len(linhas)

    # Encontrar cabeçalho do hidrograma
    indices = [
        i for i in range(inicio, fim)
        if linhas[i].startswith("Flow Hydrograph=")
    ]

    if len(indices) != 1:
        raise ValueError(
            "Flow Hydrograph não encontrado unicamente."
        )

    i = indices[0]

    quantidade = int(linhas[i].split("=", 1)[1].strip())

    if quantidade != 17:
        raise ValueError(
            f"Arquivo contém {quantidade} valores, não 17."
        )

    if not any(
        linha.strip() == "Interval=1HOUR"
        for linha in linhas[inicio:i]
    ):
        raise ValueError("Intervalo diferente de 1HOUR.")

    # O arquivo original possui 10 + 7 valores
    if i + 2 >= fim:
        raise ValueError("Bloco numérico incompleto.")

    linha1 = linhas[i + 1].rstrip("\r\n")
    linha2 = linhas[i + 2].rstrip("\r\n")

    if len(linha1) != 80 or len(linha2) != 56:
        raise ValueError(
            "Formato numérico original diferente de 10 + 7 "
            "campos de 8 caracteres. Edição cancelada."
        )

    # Verificar que as linhas originais são numéricas
    for texto, n in [(linha1, 10), (linha2, 7)]:
        for j in range(n):
            float(texto[j*8:(j+1)*8])

    # Preservar as quebras de linha de cada registro
    def quebra_original(linha):
        if linha.endswith("\r\n"):
            return "\r\n"
        if linha.endswith("\n"):
            return "\n"
        return ""

    linhas[i + 1] = (
        "".join(campos[:10]) + quebra_original(linhas[i + 1])
    )

    linhas[i + 2] = (
        "".join(campos[10:]) + quebra_original(linhas[i + 2])
    )

    # Validar valores antes da escrita
    valores_lidos = []

    for texto, n in [
        (linhas[i + 1].rstrip("\r\n"), 10),
        (linhas[i + 2].rstrip("\r\n"), 7)
    ]:
        for j in range(n):
            valores_lidos.append(
                float(texto[j*8:(j+1)*8])
            )

    if not np.allclose(valores_lidos, vazoes, atol=0.005):
        raise ValueError("Erro na validação das vazões.")

    # Backup de segurança
    backup = Path(str(caminho) + ".pre_edicao")
    shutil.copy2(caminho, backup)

    # Gravar preservando o restante do arquivo
    with open(caminho, "w", encoding="latin-1", newline="") as f:
        f.writelines(linhas)

    print("\n17 vazões atualizadas com sucesso.")
    print("Arquivo:", caminho)
    print("Backup:", backup)


# --------------------------------------------------
# 7. ATUALIZAR BC ANTES DE ABRIR O HEC-RAS E EXECUTAR
# --------------------------------------------------
# IMPORTANTE: feche o HEC-RAS e a janela Unsteady Flow Data
# antes de executar esta célula, para evitar dados em memória.
# O plano ativo deve referenciar Case_Study.u01 e possuir
# período compatível com o hidrograma (30/01/2022 05h–21h).

if EXECUTAR_HECRAS:
    if not ARQUIVO_PRJ.is_file():
        raise FileNotFoundError(ARQUIVO_PRJ)

    # Importar COM ANTES da edição para evitar alterar o .u01
    # se pywin32 não estiver funcionando.
    import win32com.client

    # Atualiza somente as 17 vazões da BC_Montante.
    atualizar_u01(ARQUIVO_U01, hidrograma_17["Vazao_m3s"].to_numpy())

    # Confirma que o arquivo não será regravado inadvertidamente.
    conteudo_antes = ARQUIVO_U01.read_bytes()
    rc = win32com.client.Dispatch(CONTROLLER_PROGID)
    projeto_aberto = False
    try:
        rc.Project_Open(str(ARQUIVO_PRJ.resolve()))
        projeto_aberto = True
        print("Projeto aberto no HEC-RAS 7.0.1.")

        if MOSTRAR_HECRAS:
            rc.ShowRAS()

        # Checagem do efeito de abrir o projeto.
        if ARQUIVO_U01.read_bytes() != conteudo_antes:
            raise RuntimeError("O .u01 foi modificado durante Project_Open. Execução interrompida.")

        print("Executando simulação HEC-RAS...")
        resultado_ras = rc.Compute_CurrentPlan(None, None, True)
        print("Retorno do Controller:", resultado_ras)

        if ARQUIVO_U01.read_bytes() != conteudo_antes:
            print("ATENÇÃO: o .u01 foi alterado durante o cálculo; confira as condições de contorno.")

        # Não chamar Project_Save(): evitar sobrescrever o .u01.
    finally:
        if FECHAR_HECRAS_AO_FINAL and projeto_aberto:
            rc.QuitRAS()
        elif projeto_aberto:
            print("HEC-RAS mantido aberto para inspeção. Feche-o manualmente ao terminar.")
else:
    print("HEC-RAS não executado; as 17 vazões foram preparadas apenas em memória.")

# %% 20 | Mapa HTML hibrido: envoltoria maxima e edificios atingidos
import folium
# Execute somente depois que os arquivos SIG do RAS estiverem disponiveis.

def mapa_base(gdf, titulo):

    base = gdf.to_crs(epsg=4326)

    centro = base.geometry.union_all().centroid

    mapa = folium.Map(
        location=[centro.y, centro.x],
        zoom_start=15,
        tiles=None,
        control_scale=True
    )

    # Google Hybrid: satélite + ruas e nomes
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Hybrid",
        overlay=False,
        control=True,
        max_zoom=21
    ).add_to(mapa)

    # Alternativa: imagens de satélite Esri
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri",
        name="Esri Satellite",
        overlay=False,
        control=True,
        max_zoom=19
    ).add_to(mapa)

    # Ajustar visualização à área de estudo
    limites = base.total_bounds

    mapa.fit_bounds([
        [limites[1], limites[0]],
        [limites[3], limites[2]]
    ])

    return mapa

if not CAMINHO_MANCHA.exists() or not CAMINHO_EDIFICIOS.exists():
    print('Configure CAMINHO_MANCHA e CAMINHO_EDIFICIOS antes da celula 20.')
else:
    mancha = gpd.read_file(CAMINHO_MANCHA)
    edificios = gpd.read_file(CAMINHO_EDIFICIOS)
    if mancha.crs is None or edificios.crs is None:
        raise ValueError('Os shapefiles precisam ter CRS definido.')
    mancha = mancha[mancha.geometry.notna() & ~mancha.geometry.is_empty].copy()
    edificios = edificios[edificios.geometry.notna() & ~edificios.geometry.is_empty].copy()
    mancha_ed = mancha.to_crs(edificios.crs)
    area_inundada = mancha_ed.geometry.union_all()
    atingidos = edificios[edificios.geometry.intersects(area_inundada)].copy()
    print('Edificacoes com intersecao com a mancha:', len(atingidos))
    # Intersecoes nao equivalem necessariamente a residencias: conferir uso do imovel.
    mapa_inundacao = mapa_base(mancha, 'Inundacao maxima')
    folium.GeoJson(mancha.to_crs(4326), name='Envoltoria maxima',
                   style_function=lambda _: {'color':'#0072B2','weight':2,
                                              'fillColor':'#0072B2','fillOpacity':0.25}).add_to(mapa_inundacao)
    if len(atingidos):
        folium.GeoJson(atingidos.to_crs(4326), name='Edificacoes atingidas',
                       style_function=lambda _: {'color':'#D55E00','weight':1,
                                                  'fillColor':'#D55E00','fillOpacity':0.6}).add_to(mapa_inundacao)
    mapa_inundacao.get_root().html.add_child(Element(
        f'<div style="position:fixed;top:15px;left:55px;z-index:9999;background:white;'
        f'padding:12px;border-radius:5px"><b>Edificacoes atingidas: {len(atingidos)}</b></div>'))
    folium.LayerControl().add_to(mapa_inundacao)
    html_inundacao = PASTA_MAPAS / '01_mapa_inundacao_edificios.html'
    mapa_inundacao.save(str(html_inundacao))
    print('Mapa salvo:', html_inundacao)
    webbrowser.open(html_inundacao.resolve().as_uri())

# %% 21 | Mapa HTML hibrido: profundidade maxima e danos por edificacao
# Funcao de dano do script de referencia: y = 90.832 + 39.334 ln(profundidade)
# y em R$/m2; dano = y * area do poligono em m2.
if 'atingidos' not in globals() or not CAMINHO_DEPTH.exists():
    print('Execute a celula 20 e configure CAMINHO_DEPTH antes da celula 21.')
else:
    danos = atingidos.copy()
    profundidades = []
    with rasterio.open(CAMINHO_DEPTH) as src:
        if src.crs is None:
            raise ValueError('Raster sem CRS.')
        edificios_raster = danos.to_crs(src.crs)
        for geom in edificios_raster.geometry:
            try:
                recorte, _ = mask(src, [mapping(geom)], crop=True,
                                  all_touched=True, filled=False)
                validos = recorte[0].compressed()
                validos = validos[np.isfinite(validos) & (validos >= 0)]
                profundidades.append(float(validos.mean()) if len(validos) else np.nan)
            except ValueError:
                profundidades.append(np.nan)
    # Calculo de area em CRS projetado metrico; nao em graus.
    if edificios_raster.crs.is_projected and edificios_raster.crs.axis_info[0].unit_name.lower() in ('metre', 'meter', 'metro'):
        area_gdf = edificios_raster
    else:
        area_gdf = danos.to_crs(danos.estimate_utm_crs())
    danos['area_m2'] = area_gdf.geometry.area.to_numpy()
    danos['depth_m'] = profundidades
    d = danos['depth_m'].to_numpy(dtype=float)
    # Salvaguarda: a formula logaritmica pode resultar em dano negativo para profundidades baixas.
    danos_m2 = np.where(np.isfinite(d) & (d > 0.001),
                        np.maximum(0, 90.832 + 39.334 * np.log(np.maximum(d, 0.001))), 0)
    danos['dano_R_m2'] = danos_m2
    danos['dano_R'] = danos['area_m2'] * danos['dano_R_m2']
    total_danos = float(danos['dano_R'].sum())
    print(f'Dano total estimado: R$ {total_danos:,.2f}')
    mapa_danos = mapa_base(mancha, 'Danos por edificacao')
    folium.GeoJson(mancha.to_crs(4326), name='Envoltoria maxima',
                   style_function=lambda _: {'color':'#0072B2','weight':1,
                                              'fillColor':'#0072B2','fillOpacity':0.13}).add_to(mapa_danos)
    if not danos.empty:
        dados_web = danos[['area_m2','depth_m','dano_R_m2','dano_R','geometry']].to_crs(4326).copy()
        folium.GeoJson(dados_web, name='Danos por edificacao',
                       style_function=lambda f: {'color':'#7f1d1d','weight':0.7,
                                                  'fillColor':'#d94801','fillOpacity':0.65},
                       tooltip=folium.GeoJsonTooltip(
                           fields=['area_m2','depth_m','dano_R_m2','dano_R'],
                           aliases=['Area (m2)','Profundidade media (m)',
                                    'Dano (R$/m2)','Dano total (R$)'], localize=True)).add_to(mapa_danos)
    mapa_danos.get_root().html.add_child(Element(
        f'<div style="position:fixed;top:15px;left:55px;z-index:9999;background:white;'
        f'padding:12px;border-radius:5px"><b>Danos estimados: R$ {total_danos:,.2f}</b></div>'))
    folium.LayerControl().add_to(mapa_danos)
    html_danos = PASTA_MAPAS / '02_mapa_danos_edificacoes.html'
    mapa_danos.save(str(html_danos))
    print('Mapa salvo:', html_danos)
    webbrowser.open(html_danos.resolve().as_uri())