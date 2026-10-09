# -*- coding: utf-8 -*-
"""HydroUAI | Estudo de caso 1 — LSTM recursiva e potencial energético.

No Spyder, execute as células (# %%) EM ORDEM com Ctrl+Enter.
Coloque series_preenchidas.csv na mesma pasta deste arquivo.
Gráficos são exibidos no Spyder; tabelas são exportadas em CSV.
"""

# %% 01 | Bibliotecas e configurações
from pathlib import Path
from itertools import product
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

PASTA = Path(__file__).resolve().parent
ARQUIVO = PASTA / 'series_preenchidas.csv'
SAIDA = PASTA / 'resultados'
SAIDA.mkdir(parents=True, exist_ok=True)
ALVO = 'Q_Afluente'

JANELAS = [1,3,5]
CAMADAS_GRID = [1,2]
HIDDEN_GRID = [32,64]
USAR_EARLY_STOPPING = False
USAR_LR_ADAPTATIVO = False
EPOCAS_MAX = 300
PACIENCIA_EARLY = 18
PACIENCIA_LR = 5
LR_INICIAL = 0.005
BATCH_SIZE = 128
SEMENTE = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Dispositivo:', DEVICE)
print('Combinações:', len(JANELAS)*len(CAMADAS_GRID)*len(HIDDEN_GRID))

# %% 02 | Ler a série diária de vazão
if not ARQUIVO.exists():
    raise FileNotFoundError(f'Arquivo não encontrado: {ARQUIVO}')
df = pd.read_csv(ARQUIVO, encoding='utf-8-sig')
df['data'] = pd.to_datetime(df['data'], format='%Y-%m-%d')
df = df.sort_values('data').set_index('data')
if df.index.has_duplicates or not df.index.equals(
        pd.date_range(df.index.min(), df.index.max(), freq='D', name='data')):
    raise ValueError('Há datas duplicadas ou lacunas no calendário diário.')
if ALVO not in df:
    raise KeyError(f'Coluna ausente: {ALVO}')
datas = df.index
q = pd.to_numeric(df[ALVO], errors='coerce').to_numpy(dtype=float, copy=True)
q[~np.isfinite(q) | (q < 0)] = np.nan
print(f'Período: {datas.min().date()} a {datas.max().date()}')
print(f'Dias: {len(q)} | Valores válidos: {np.isfinite(q).sum()}')
print(df[[ALVO]].head())

# %% 03 | Separar treino (70%), validação (15%) e teste (15%)
n = len(q)
corte1, corte2 = int(0.70*n), int(0.85*n)
blocos = np.zeros(n, dtype=int)
blocos[corte1:corte2] = 1
blocos[corte2:] = 2
for k, nome in enumerate(['Treino', 'Validação', 'Teste']):
    ids = np.flatnonzero(blocos == k)
    print(f'{nome}: {len(ids)} dias | {datas[ids[0]].date()} a {datas[ids[-1]].date()}')

# %% 04 | Visualizar a série e os três períodos
cores = ['#0072B2', '#E69F00', '#009E73']
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(datas, q, color='#333333', lw=0.7, label='Vazão observada', zorder=2)
for k, nome in enumerate(['Treino (70%)', 'Validação (15%)', 'Teste (15%)']):
    ids = np.flatnonzero(blocos == k)
    ax.axvspan(datas[ids[0]], datas[ids[-1]], facecolor=cores[k],
               edgecolor=cores[k], alpha=0.11, hatch='///', label=nome)
    if k:
        ax.axvline(datas[ids[0]], color=cores[k], ls='--', lw=1)
ax.set(title='Série diária e divisão cronológica', xlabel='Data', ylabel='Vazão (m³/s)')
ax.grid(alpha=0.2)
ax.legend(ncol=4, fontsize=9)
fig.tight_layout()
plt.show()

# %% 05 | Normalizar usando SOMENTE o treino
scaler = StandardScaler()
q_treino = q[:corte1]
scaler.fit(q_treino[np.isfinite(q_treino)].reshape(-1, 1))
q_norm = np.full(len(q), np.nan, dtype=float)
validos = np.isfinite(q)
q_norm[validos] = scaler.transform(q[validos].reshape(-1, 1)).ravel()
print('Média do treino:', round(float(scaler.mean_[0]), 3))
print('Desvio-padrão do treino:', round(float(scaler.scale_[0]), 3))

# %% 06 | Construir janelas antecedentes e alvos T+1, T+2, T+3
def montar_janelas(H):
    """Retorna três conjuntos; cada origem e seus três alvos ficam no mesmo bloco."""
    origens = []
    for t in range(H-1, len(q)-3):
        if blocos[t] == blocos[t+3] and np.isfinite(q[t-H+1:t+4]).all():
            origens.append(t)
    origens = np.asarray(origens, dtype=int)
    if not len(origens):
        raise ValueError(f'Sem janelas válidas para H={H}')
    x = np.stack([q_norm[t-H+1:t+1] for t in origens]).astype('float32')[:, :, None]
    y = np.stack([q_norm[t+1:t+4] for t in origens]).astype('float32')
    conjuntos = {}
    for k, nome in enumerate(['treino', 'validacao', 'teste']):
        m = blocos[origens] == k
        if m.sum() < 10:
            raise ValueError(f'Menos de 10 exemplos no bloco {nome}')
        conjuntos[nome] = (torch.from_numpy(x[m]), torch.from_numpy(y[m]), origens[m])
    return conjuntos

# Exemplo para explicar as dimensões antes do Grid Search
exemplo_H = 3
exemplo = montar_janelas(exemplo_H)
for nome, (x, y, origens) in exemplo.items():
    print(f'{nome}: X={tuple(x.shape)} | Y={tuple(y.shape)}')
print('X: amostras × dias antecedentes × 1 variável; Y: T+1, T+2, T+3')

# %% 07 | Definir uma LSTM para prever APENAS o próximo dia
class ModeloLSTM(nn.Module):
    def __init__(self, camadas, hidden_units):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_units,
                            num_layers=camadas, batch_first=True)
        self.saida = nn.Linear(hidden_units, 1)

    def forward(self, x):
        saida, _ = self.lstm(x)
        return self.saida(saida[:, -1, :])

# %% 08 | Aplicar a mesma LSTM recursivamente por três dias
@torch.no_grad()
def prever_recursivo(modelo, x, passos=3):
    modelo.eval()
    lotes = []
    for inicio in range(0, len(x), 1024):
        janela = x[inicio:inicio+1024].to(DEVICE).clone()
        previsoes = []
        for passo in range(passos):
            proxima_vazao = modelo(janela)
            previsoes.append(proxima_vazao)
            # Descarta o dia mais antigo e incorpora a previsão recém-gerada.
            janela = torch.cat((janela[:, 1:, :], proxima_vazao.unsqueeze(1)), dim=1)
        lotes.append(torch.cat(previsoes, dim=1).cpu().numpy())
    return np.concatenate(lotes, axis=0)

# %% 09 | Treinar um candidato e registrar a evolução das perdas
def treinar(conjuntos, camadas, hidden_units):
    random.seed(SEMENTE)
    np.random.seed(SEMENTE)
    torch.manual_seed(SEMENTE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEMENTE)
    modelo = ModeloLSTM(camadas, hidden_units).to(DEVICE)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=LR_INICIAL)
    scheduler = None
    if USAR_LR_ADAPTATIVO:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            otimizador, mode='min', factor=0.5, patience=PACIENCIA_LR, min_lr=1e-6)

    x_tr, y_tr, _ = conjuntos['treino']
    x_val, y_val, _ = conjuntos['validacao']
    loader = DataLoader(TensorDataset(x_tr, y_tr[:, :1]), batch_size=BATCH_SIZE,
                        shuffle=True, generator=torch.Generator().manual_seed(SEMENTE))
    historico = []
    melhor_mse, melhor_epoca, espera = float('inf'), 0, 0
    melhores_pesos = copy.deepcopy(modelo.state_dict())

    for epoca in range(1, EPOCAS_MAX+1):
        modelo.train()
        soma = 0.0
        lr_atual = otimizador.param_groups[0]['lr']
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            otimizador.zero_grad()
            erro = nn.functional.mse_loss(modelo(xb), yb)
            erro.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            otimizador.step()
            soma += erro.item()*len(xb)

        previsao_val = prever_recursivo(modelo, x_val)
        y_val_np = y_val.numpy()
        mse_val_T1 = float(np.mean((previsao_val[:, 0] - y_val_np[:, 0])**2))
        mse_val_rec = float(np.mean((previsao_val - y_val_np)**2))
        if scheduler is not None:
            scheduler.step(mse_val_rec)
        historico.append(dict(epoca=epoca, mse_treino_T1=soma/len(x_tr),
                              mse_validacao_T1=mse_val_T1,
                              mse_validacao_recursiva=mse_val_rec, learning_rate=lr_atual))
        if mse_val_rec < melhor_mse - 1e-6:
            melhor_mse, melhor_epoca, espera = mse_val_rec, epoca, 0
            melhores_pesos = copy.deepcopy(modelo.state_dict())
        else:
            espera += 1
        if epoca == 1 or epoca % 20 == 0:
            print(f'Época {epoca:3d} | treino T+1={soma/len(x_tr):.4f} | '
                  f'validação T+1={mse_val_T1:.4f} | validação recursiva={mse_val_rec:.4f}')
        if USAR_EARLY_STOPPING and espera >= PACIENCIA_EARLY:
            print(f'Early stopping na época {epoca}; restaurando época {melhor_epoca}.')
            break

    if USAR_EARLY_STOPPING:
        modelo.load_state_dict(melhores_pesos)
    # Sem early stopping: modelo da ÚLTIMA época, inclusive para demonstrar overfitting.
    mse_selecao = float(np.mean((prever_recursivo(modelo, x_val)-y_val.numpy())**2))
    return modelo, pd.DataFrame(historico), mse_selecao, melhor_epoca

# %% 10 | Grid Search: H × camadas × hidden units
resultados_grid = []
melhor_mse = float('inf')
melhor_modelo = None
melhor_conjuntos = None
melhor_config = None
melhor_historico = None

for H, camadas, hidden in product(JANELAS, CAMADAS_GRID, HIDDEN_GRID):
    print(f'\n=== H={H} | camadas={camadas} | unidades={hidden} ===')
    conjuntos = montar_janelas(H)
    modelo, historico, mse_val, melhor_epoca = treinar(conjuntos, camadas, hidden)
    identificador = f'H{H}_L{camadas}_U{hidden}'
    historico.to_csv(SAIDA/f'historico_{identificador}.csv', index=False)
    resultados_grid.append(dict(H=H, camadas=camadas, hidden_units=hidden,
                                MSE_validacao_recursiva=mse_val,
                                melhor_epoca_validacao=melhor_epoca,
                                epocas_executadas=len(historico)))
    if mse_val < melhor_mse:
        melhor_mse = mse_val
        melhor_config = (H, camadas, hidden)
        melhor_modelo = modelo
        melhor_conjuntos = conjuntos
        melhor_historico = historico.copy()

ranking = pd.DataFrame(resultados_grid).sort_values('MSE_validacao_recursiva')
ranking.to_csv(SAIDA/'ranking_grid_search.csv', index=False)
print('\nRanking de validação:\n', ranking.to_string(index=False))
print('\nMelhor configuração:', melhor_config)

# %% 11 | Examinar as curvas de aprendizado do vencedor
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(melhor_historico.epoca, melhor_historico.mse_treino_T1,
        label='Treino T+1', color='#0072B2')
ax.plot(melhor_historico.epoca, melhor_historico.mse_validacao_T1,
        label='Validação T+1 (comparável ao treino)', color='#D55E00')
ax.plot(melhor_historico.epoca, melhor_historico.mse_validacao_recursiva,
        label='Validação recursiva T+1 a T+3', color='#009E73', alpha=0.8)
melhor_epoca = int(melhor_historico.loc[
    melhor_historico.mse_validacao_recursiva.idxmin(), 'epoca'])
ax.axvline(melhor_epoca, color='gray', ls='--', label='Melhor época de validação')
ax.set(xlabel='Época', ylabel='MSE normalizado',
       title=f'Aprendizado do vencedor | H={melhor_config[0]}, '
             f'L={melhor_config[1]}, U={melhor_config[2]}')
ax.grid(alpha=0.2)
ax.legend()
fig.tight_layout()
plt.show()

# %% 12 | Gerar previsões de VAZÃO para treino, validação e teste
modelo = melhor_modelo
conjuntos = melhor_conjuntos
tabelas = []
for nome, (x, y, origens) in conjuntos.items():
    pred_norm = prever_recursivo(modelo, x)
    obs = scaler.inverse_transform(y.numpy().reshape(-1, 1)).reshape(-1, 3)
    pred = scaler.inverse_transform(pred_norm.reshape(-1, 1)).reshape(-1, 3)
    pred = np.maximum(pred, 0)
    for h in [1, 2, 3]:
        tabelas.append(pd.DataFrame({
            'bloco': nome, 'data_origem': datas[origens],
            'data_alvo': datas[origens+h], 'horizonte_dias': h,
            'Q_observada': obs[:, h-1], 'Q_prevista': pred[:, h-1]}))
previsoes = pd.concat(tabelas, ignore_index=True)
previsoes.to_csv(SAIDA/'previsoes_vazao.csv', index=False)
print(previsoes.head())

# %% 13 | Avaliar a qualidade das previsões de VAZÃO
def metricas(obs, pred):
    obs, pred = np.asarray(obs), np.asarray(pred)
    erro = pred - obs
    den = np.sum((obs - obs.mean())**2)
    r = np.corrcoef(obs, pred)[0, 1] if obs.std()>0 and pred.std()>0 else np.nan
    alpha = pred.std()/obs.std() if obs.std()>0 else np.nan
    beta = pred.mean()/obs.mean() if obs.mean()!=0 else np.nan
    return dict(RMSE=np.sqrt(np.mean(erro**2)), MAE=np.mean(np.abs(erro)),
                NSE=1-np.sum(erro**2)/den if den>0 else np.nan,
                KGE=1-np.sqrt((r-1)**2+(alpha-1)**2+(beta-1)**2),
                PBIAS_pct=100*np.sum(erro)/np.sum(obs) if np.sum(obs)!=0 else np.nan)

resumo_q = []
for (nome, h), tab in previsoes.groupby(['bloco', 'horizonte_dias'], sort=False):
    resumo_q.append(dict(bloco=nome, horizonte_dias=h,
                         **metricas(tab.Q_observada, tab.Q_prevista)))
metricas_q = pd.DataFrame(resumo_q)
metricas_q.to_csv(SAIDA/'metricas_vazao.csv', index=False)
print('\nMétricas de vazão no teste:\n',
      metricas_q.query('bloco == "teste"').round(3).to_string(index=False))

# %% 14 | Gráfico final da etapa hidrológica: vazões no TESTE
fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
for h, ax in zip([1, 2, 3], axes):
    tab = previsoes.query('bloco == "teste" and horizonte_dias == @h')
    ax.plot(tab.data_alvo, tab.Q_observada, label='Observada', color='#0072B2', lw=0.9)
    ax.plot(tab.data_alvo, tab.Q_prevista, label='LSTM recursiva', color='#D55E00', lw=0.9)
    ax.set(title=f'Vazão no teste | T+{h}', ylabel='Vazão (m³/s)')
    ax.grid(alpha=0.2)
    ax.legend()
plt.show()

# %% 15 | Aplicação: potencial energético equivalente

# UHE Três Marias — PAE, revisão F (2024)
# Referências: páginas 7 e 46

# Níveis de referência documentados no PAE (m)
NA_MONTANTE = 572.50
NA_JUSANTE = 515.70

# Queda bruta aproximada (m)
H_REF_M = NA_MONTANTE - NA_JUSANTE

# Rendimento global hipotético (não informado no PAE)
RENDIMENTO = 0.90

# Potência instalada informada no PAE (MW)
P_INST_MW = 396.0

def converter_energia(q_valores):
    potencia = 0.00981 * RENDIMENTO * H_REF_M * np.maximum(q_valores, 0)
    potencia = np.minimum(potencia, P_INST_MW)
    energia = potencia * 24  # MWh do dia-alvo, NÃO energia acumulada até T+h
    return potencia, energia

previsoes['P_referencia_MW'], previsoes['E_referencia_MWh'] = converter_energia(
    previsoes.Q_observada.to_numpy())
previsoes['P_prevista_MW'], previsoes['E_prevista_MWh'] = converter_energia(
    previsoes.Q_prevista.to_numpy())
previsoes.to_csv(SAIDA/'previsoes_vazao_energia.csv', index=False)

# %% 16 | ÚLTIMA FIGURA: energia equivalente observada e prevista
fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
for h, ax in zip([1, 2, 3], axes):
    tab = previsoes.query('bloco == "teste" and horizonte_dias == @h')
    ax.plot(tab.data_alvo, tab.E_referencia_MWh,
            label='Energia equivalente de referência', color='#0072B2', lw=0.9)
    ax.plot(tab.data_alvo, tab.E_prevista_MWh,
            label='Energia equivalente prevista', color='#D55E00', lw=0.9)
    ax.set(title=f'Energia equivalente no teste | T+{h}', ylabel='MWh/dia')
    ax.grid(alpha=0.2)
    ax.legend()
plt.show()
print('\nResultados em:', SAIDA)
