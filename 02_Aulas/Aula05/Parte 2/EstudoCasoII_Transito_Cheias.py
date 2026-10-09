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
ESTACOES_P = None  # None: todas as colunas P_; ou lista de estações
HORIZONTES = 5  # 7 modelos independentes: T+1, ..., T+7 DIAS

JANELAS = [ 5, 7]
CAMADAS_GRID = [1, 2]
HIDDEN_GRID = [ 64,128 ]
USAR_EARLY_STOPPING = False
USAR_LR_ADAPTATIVO = False
EPOCAS_MAX = 50
PACIENCIA_EARLY = 25
PACIENCIA_LR = 5
LR_INICIAL = 0.001
BATCH_SIZE = 128
SEMENTE = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Dispositivo:', DEVICE)
print('Modelos finais:', HORIZONTES, '| candidatos por horizonte:',
      len(JANELAS) * len(CAMADAS_GRID) * len(HIDDEN_GRID))

# %% 02 | Ler a série diária de vazão e precipitação
if not ARQUIVO.exists():
    raise FileNotFoundError(f'Arquivo não encontrado: {ARQUIVO}')
df = pd.read_csv(ARQUIVO, encoding='utf-8-sig')
FORMATO_DATA = '%m/%d/%Y'
df['data'] = pd.to_datetime(df['data'], format=FORMATO_DATA)
df = df.sort_values('data').set_index('data')
if df.index.has_duplicates or not df.index.equals(
        pd.date_range(df.index.min(), df.index.max(), freq='D', name='data')):
    raise ValueError('Há datas duplicadas ou lacunas no calendário diário.')
if ALVO not in df.columns:
    raise KeyError(f'Coluna ausente: {ALVO}')
COLUNAS_P = ([c for c in df.columns if c.startswith('P_')]
             if ESTACOES_P is None else list(ESTACOES_P))
if not COLUNAS_P:
    raise ValueError('Não foram encontradas colunas P_. Defina ESTACOES_P.')
if any(c not in df.columns for c in COLUNAS_P):
    raise KeyError(f'Estações ausentes: {set(COLUNAS_P) - set(df.columns)}')
COLUNAS_ENTRADA = [ALVO] + COLUNAS_P
dados = df[COLUNAS_ENTRADA].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float, copy=True)
dados[~np.isfinite(dados) | (dados < 0)] = np.nan
q, p = dados[:, 0], dados[:, 1:]
datas = df.index
print(f'Período: {datas.min().date()} a {datas.max().date()}')
print('Entradas (somente antecedentes):', COLUNAS_ENTRADA)
print(df[COLUNAS_ENTRADA].head())

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
ax.plot(datas, q, color='#333333', lw=0.7, label='Vazão observada')
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

# %% 05 | Normalizar Q e P utilizando SOMENTE o treino
scaler_q = StandardScaler()
scaler_p = StandardScaler()
q_treino = q[:corte1]
scaler_q.fit(q_treino[np.isfinite(q_treino)].reshape(-1, 1))
scaler_p.fit(p[:corte1])
q_norm = scaler_q.transform(q.reshape(-1, 1)).ravel()
p_norm = scaler_p.transform(p)
dados_norm = np.column_stack([q_norm, p_norm])
print('Número de variáveis:', dados_norm.shape[1])

# %% 06 | Montar entradas Q+P passadas e UM alvo específico T+h
# Em cada amostra: X = dias [t-H+1, ..., t]; Y = vazão em t+h.
# Nenhuma chuva ou vazão observada após t entra em X.
def montar_janelas(H, horizonte):
    origens = []
    for t in range(H-1, n-horizonte):
        if blocos[t] != blocos[t+horizonte]:
            continue
        if not np.isfinite(dados_norm[t-H+1:t+1]).all():
            continue
        if not np.isfinite(q_norm[t+horizonte]):
            continue
        origens.append(t)
    origens = np.asarray(origens, dtype=int)
    if len(origens) == 0:
        raise ValueError(f'Sem amostras válidas: H={H}, horizonte={horizonte}')
    x = np.stack([dados_norm[t-H+1:t+1] for t in origens]).astype('float32')
    y = q_norm[origens+horizonte].astype('float32')[:, None]
    conjuntos = {}
    for k, nome in enumerate(['treino', 'validacao', 'teste']):
        m = blocos[origens] == k
        if m.sum() < 10:
            raise ValueError(f'Menos de 10 amostras em {nome} para T+{horizonte}, H={H}')
        conjuntos[nome] = (torch.from_numpy(x[m]), torch.from_numpy(y[m]), origens[m])
    return conjuntos

exemplo = montar_janelas(JANELAS[0], 1)
for nome, (x, y, origens) in exemplo.items():
    print(f'{nome}: X={tuple(x.shape)} | Y={tuple(y.shape)}')
print('X: amostras × dias antecedentes × (Q + estações P); Y: uma vazão futura')

# %% 07 | Modelo LSTM: uma saída escalar por horizonte
class ModeloLSTM(nn.Module):
    def __init__(self, camadas, hidden_units):
        super().__init__()
        self.lstm = nn.LSTM(input_size=len(COLUNAS_ENTRADA), hidden_size=hidden_units,
                            num_layers=camadas, batch_first=True)
        self.saida = nn.Linear(hidden_units, 1)

    def forward(self, x):
        saida, _ = self.lstm(x)
        return self.saida(saida[:, -1, :])

# %% 08 | Prever diretamente, SEM realimentação e SEM chuva futura
@torch.no_grad()
def prever_direto(modelo, x):
    modelo.eval()
    partes = []
    for inicio in range(0, len(x), 1024):
        partes.append(modelo(x[inicio:inicio+1024].to(DEVICE)).cpu().numpy())
    return np.concatenate(partes, axis=0)

# %% 09 | Treinar um candidato para um horizonte específico
# Backpropagation, early stopping e LR adaptativo: MSE de T+h.
def treinar(conjuntos, camadas, hidden_units, horizonte):
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
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=BATCH_SIZE,
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
        pred_val = prever_direto(modelo, x_val)
        mse_val = float(np.mean((pred_val - y_val.numpy())**2))
        if scheduler is not None:
            scheduler.step(mse_val)
        historico.append(dict(epoca=epoca, mse_treino=soma/len(x_tr),
                              mse_validacao=mse_val, learning_rate=lr_atual))
        if mse_val < melhor_mse - 1e-6:
            melhor_mse, melhor_epoca, espera = mse_val, epoca, 0
            melhores_pesos = copy.deepcopy(modelo.state_dict())
        else:
            espera += 1
        if epoca == 1 or epoca % 20 == 0:
            print(f'T+{horizonte} | época {epoca:3d} | treino={soma/len(x_tr):.4f} '
                  f'| validação={mse_val:.4f}')
        if USAR_EARLY_STOPPING and espera >= PACIENCIA_EARLY:
            print(f'Early stopping T+{horizonte}: época {epoca}; melhor={melhor_epoca}')
            break
    if USAR_EARLY_STOPPING:
        modelo.load_state_dict(melhores_pesos)
    mse_selecao = float(np.mean((prever_direto(modelo, x_val)-y_val.numpy())**2))
    return modelo, pd.DataFrame(historico), mse_selecao, melhor_epoca

# %% 10 | Grid Search INDEPENDENTE para cada horizonte T+1,...,T+7
resultados_grid = []
melhores_modelos = {}
melhores_conjuntos = {}
melhores_configs = {}
melhores_historicos = {}
for horizonte in range(1, HORIZONTES+1):
    print(f'\n========== TREINAMENTO DIRETO T+{horizonte} ==========')
    melhor_mse = float('inf')
    for H, camadas, hidden in product(JANELAS, CAMADAS_GRID, HIDDEN_GRID):
        print(f'\nT+{horizonte} | H={H} | camadas={camadas} | unidades={hidden}')
        conjuntos = montar_janelas(H, horizonte)
        modelo, historico, mse_val, melhor_epoca = treinar(conjuntos, camadas, hidden, horizonte)
        identificador = f'T{horizonte}_H{H}_L{camadas}_U{hidden}'
        historico.to_csv(SAIDA/f'historico_{identificador}.csv', index=False)
        resultados_grid.append(dict(horizonte_dias=horizonte, H=H, camadas=camadas,
                                    hidden_units=hidden, MSE_validacao=mse_val,
                                    melhor_epoca_validacao=melhor_epoca,
                                    epocas_executadas=len(historico)))
        if mse_val < melhor_mse:
            melhor_mse = mse_val
            melhores_modelos[horizonte] = modelo
            melhores_conjuntos[horizonte] = conjuntos
            melhores_configs[horizonte] = (H, camadas, hidden)
            melhores_historicos[horizonte] = historico.copy()
    print(f'Melhor T+{horizonte}: {melhores_configs[horizonte]} | MSE={melhor_mse:.5f}')
ranking = pd.DataFrame(resultados_grid).sort_values(['horizonte_dias', 'MSE_validacao'])
ranking.to_csv(SAIDA/'ranking_grid_search.csv', index=False)
melhores_tabela = ranking.groupby('horizonte_dias', as_index=False).first()
melhores_tabela.to_csv(SAIDA/'melhores_por_horizonte.csv', index=False)
print('\nMelhores configurações por horizonte:\n', melhores_tabela.to_string(index=False))

# %% 11 | Curvas de aprendizado: uma figura para cada horizonte
fig, axes = plt.subplots(HORIZONTES, 1, figsize=(11, 17), constrained_layout=True)
for horizonte, ax in zip(range(1, HORIZONTES+1), axes):
    hist = melhores_historicos[horizonte]
    ax.plot(hist.epoca, hist.mse_treino, label='Treino T+h', color='#0072B2')
    ax.plot(hist.epoca, hist.mse_validacao, label='Validação T+h', color='#D55E00')
    epoca_melhor = int(hist.loc[hist.mse_validacao.idxmin(), 'epoca'])
    ax.axvline(epoca_melhor, color='gray', ls='--', lw=1)
    ax.set(title=f'T+{horizonte} | H, camadas, unidades = {melhores_configs[horizonte]}',
           ylabel='MSE normalizado')
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
axes[-1].set_xlabel('Época')
plt.show()

# %% 12 | Previsões DIRETAS de vazão para treino, validação e teste
# Cada T+h usa seu próprio modelo, sua própria janela e a MESMA data de origem.
tabelas = []
for horizonte in range(1, HORIZONTES+1):
    modelo = melhores_modelos[horizonte]
    for nome, (x, y, origens) in melhores_conjuntos[horizonte].items():
        pred_norm = prever_direto(modelo, x)
        obs = scaler_q.inverse_transform(y.numpy()).ravel()
        pred = scaler_q.inverse_transform(pred_norm).ravel()
        pred = np.maximum(pred, 0)
        tabelas.append(pd.DataFrame({
            'bloco': nome, 'data_origem': datas[origens],
            'data_alvo': datas[origens+horizonte], 'horizonte_dias': horizonte,
            'Q_observada': obs, 'Q_prevista': pred}))
previsoes = pd.concat(tabelas, ignore_index=True)
previsoes.to_csv(SAIDA/'previsoes_vazao.csv', index=False)
print(previsoes.head())

# %% 13 | Avaliar a qualidade por horizonte
# Cada horizonte é avaliado independentemente no teste.
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
print('\nMétricas no teste:\n',
      metricas_q.query('bloco == "teste"').round(3).to_string(index=False))

# %% 14 | Séries de teste por horizonte (origens móveis)
fig, axes = plt.subplots(HORIZONTES, 1, figsize=(12, 15), constrained_layout=True)
for h, ax in zip(range(1, HORIZONTES+1), axes):
    tab = previsoes.query('bloco == "teste" and horizonte_dias == @h')
    ax.plot(tab.data_alvo, tab.Q_observada, label='Observada', color='#0072B2', lw=0.9)
    ax.plot(tab.data_alvo, tab.Q_prevista, label=f'LSTM direta T+{h}', color='#D55E00', lw=0.9)
    ax.set(title=f'Teste | previsão direta T+{h}', ylabel='Vazão (m³/s)')
    ax.grid(alpha=0.2)
    ax.legend()
plt.show()

# %% 15 | Selecionar UMA data de origem comum aos sete modelos
DATA_ORIGEM = '2020-01-24'  # ou None para origem com maior pico previsto
COTA_INICIAL_M = 550.0
DT_HORAS = 24.0
# Como cada horizonte pode ter uma janela H diferente, usar apenas origens
# para as quais TODOS os sete modelos geraram previsão.
teste = previsoes.loc[previsoes.bloco == 'teste'].copy()
contagem = teste.groupby('data_origem').horizonte_dias.nunique()
origens_completas = contagem.index[contagem == HORIZONTES]
if len(origens_completas) == 0:
    raise ValueError('Nenhuma data de origem tem os sete horizontes disponíveis.')
if DATA_ORIGEM is None:
    picos = teste.loc[teste.data_origem.isin(origens_completas)].groupby('data_origem').Q_prevista.max()
    data_escolhida = picos.idxmax()
else:
    data_escolhida = pd.Timestamp(DATA_ORIGEM)
    if data_escolhida not in origens_completas:
        raise ValueError('DATA_ORIGEM não possui os sete horizontes no teste. '
                         f'Faixa disponível: {origens_completas.min().date()} '
                         f'a {origens_completas.max().date()}')
hidrograma = teste.loc[teste.data_origem == data_escolhida].sort_values('horizonte_dias').copy()
assert len(hidrograma) == HORIZONTES
q_afluente = hidrograma.Q_prevista.to_numpy(dtype=float)
q_observada = hidrograma.Q_observada.to_numpy(dtype=float)
q_inicial = float(df.loc[data_escolhida, ALVO])
print('Origem:', data_escolhida.date())
print('Cota inicial:', COTA_INICIAL_M, 'm')
print('Previsões diretas independentes (m³/s):', np.round(q_afluente, 1))
print('Entradas usadas: somente Q e P até a data de origem; SEM chuva futura.')

# %% 16 | Hidrograma previsto no contexto da série histórica

# Quantidade de dias anteriores à origem exibidos no gráfico
DIAS_ANTERIORES = 10

# Definir intervalo histórico até a data de origem
data_inicio = data_escolhida - pd.Timedelta(days=DIAS_ANTERIORES)

# Série histórica observada (somente até a origem)
historico = df.loc[data_inicio:data_escolhida, ALVO]

# Datas dos sete horizontes previstos
datas_futuras = pd.date_range(
    start=data_escolhida + pd.Timedelta(days=1),
    periods=HORIZONTES,
    freq='D'
)

# Incluir a vazão inicial para conectar as curvas
datas_previsao = pd.DatetimeIndex(
    [data_escolhida, *datas_futuras]
)

vazao_prevista = np.r_[q_inicial, q_afluente]
vazao_observada = np.r_[q_inicial, q_observada]

# Construir figura
fig, ax = plt.subplots(figsize=(13, 5))

# Série histórica observada até a origem
ax.plot(
    historico.index,
    historico.values,
    color='#0072B2',
    lw=1.5,
    label='Vazão observada'
)

# Continuação observada nos sete dias futuros
ax.plot(
    datas_previsao,
    vazao_observada,
    color='#0072B2',
    lw=1.8
)

# Previsão das sete LSTMs independentes
ax.plot(
    datas_previsao,
    vazao_prevista,
    color='#D55E00',
    lw=2,
    marker='o',
    markersize=4,
    label='Vazão prevista (LSTM)'
)

# Indicar a data de emissão da previsão
ax.axvline(
    data_escolhida,
    color='gray',
    linestyle='--',
    lw=1.2,
    label='Origem da previsão'
)

# Destacar discretamente o período previsto
ax.axvspan(
    data_escolhida,
    datas_futuras[-1],
    color='gray',
    alpha=0.08
)

# Limitar o gráfico até o último horizonte
ax.set_xlim(data_inicio, datas_futuras[-1])

# Eixo vertical começando em zero
ax.set_ylim(0,5000)

ax.set(
    xlabel='Data',
    ylabel='Vazão (m³/s)',
    title=f'Previsão de vazões | Origem: {data_escolhida:%d/%m/%Y}'
)

ax.grid(alpha=0.2)
ax.legend(loc='upper left')

fig.tight_layout()
plt.show()
# %% 17 | Ler as curvas características do reservatório
# cd.txt: Cota; Descarga(m3/s)
# cv.txt: Cota; Volume (1000m3)
# Atenção: descarga deve representar a estrutura/condição hidráulica simulada.
def localizar_arquivo(nome):
    for caminho in [PASTA/nome, PASTA/'05. Arquivos de Texto'/nome]:
        if caminho.exists():
            return caminho
    raise FileNotFoundError(f'Não encontrei {nome} na pasta do script nem em 05. Arquivos de Texto')

cd = pd.read_csv(localizar_arquivo('cd.txt'), sep='\t')
cv = pd.read_csv(localizar_arquivo('cv.txt'), sep='\t')
print('Colunas CD:', cd.columns.tolist())
print('Colunas CV:', cv.columns.tolist())
print(cd.head())
print(cv.head())

# %% 18 | Preparar curvas cota-volume e cota-descarga
# Usamos interpolações lineares, SEM extrapolação fora das curvas.
# Volume (1000 m³) -> m³, conforme cabeçalho do arquivo original.
cotas_cv = cv['Cota'].to_numpy(dtype=float)
volumes_m3 = cv['Volume (1000m3)'].to_numpy(dtype=float) * 1000.0
cotas_cd = cd['Cota'].to_numpy(dtype=float)
descargas = cd['Descarga(m3/s)'].to_numpy(dtype=float)

ord_cv = np.argsort(cotas_cv)
ord_cd = np.argsort(cotas_cd)
cotas_cv, volumes_m3 = cotas_cv[ord_cv], volumes_m3[ord_cv]
cotas_cd, descargas = cotas_cd[ord_cd], descargas[ord_cd]
if (np.diff(cotas_cv) <= 0).any() or (np.diff(volumes_m3) <= 0).any():
    raise ValueError('Curva cota-volume deve ser estritamente crescente.')
if (np.diff(cotas_cd) <= 0).any() or (np.diff(descargas) < 0).any():
    raise ValueError('Curva cota-descarga deve ter cotas crescentes e descargas não decrescentes.')

cota_min = max(cotas_cv.min(), cotas_cd.min())
cota_max = min(cotas_cv.max(), cotas_cd.max())
if not cota_min <= COTA_INICIAL_M <= cota_max:
    raise ValueError(f'Cota inicial fora da faixa comum das curvas: {cota_min:.2f} a {cota_max:.2f} m')

cotas_comuns = np.unique(np.r_[cotas_cv[(cotas_cv >= cota_min) & (cotas_cv <= cota_max)],
                               cotas_cd[(cotas_cd >= cota_min) & (cotas_cd <= cota_max)],
                               cota_min, cota_max])
vol_curva = np.interp(cotas_comuns, cotas_cv, volumes_m3)
q_curva = np.interp(cotas_comuns, cotas_cd, descargas)

fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
axs[0].plot(vol_curva/1e6, cotas_comuns, color='#0072B2')
axs[0].set(xlabel='Armazenamento (hm³)', ylabel='Cota (m)', title='Curva cota-volume')
axs[1].plot(q_curva, cotas_comuns, color='#009E73')
axs[1].set(xlabel='Descarga (m³/s)', ylabel='Cota (m)', title='Curva cota-descarga')
for ax in axs: ax.grid(alpha=0.2)
plt.show()

# %% 19 | Método de Puls modificado: continuidade + armazenamento
# (S2 - S1)/dt = (I1 + I2)/2 - (O1 + O2)/2
# Logo: (2*S2/dt + O2) = I1 + I2 + (2*S1/dt - O1)
# Resolvemos a relação armazenamento-descarga IMPLICITAMENTE.
def puls_modificado(afluencias_futuras, afluencia_inicial, cota_inicial, dt_horas=24):
    dt = dt_horas * 3600.0
    if dt <= 0:
        raise ValueError('dt_horas deve ser positivo.')
    if not cota_min <= cota_inicial <= cota_max:
        raise ValueError('Cota inicial fora da faixa comum das curvas.')
    s0 = np.interp(cota_inicial, cotas_comuns, vol_curva)
    o0 = np.interp(cota_inicial, cotas_comuns, q_curva)
    s, o, z = [s0], [o0], [cota_inicial]
    entradas = np.r_[float(afluencia_inicial), np.asarray(afluencias_futuras, dtype=float)]
    func_armazenamento = 2.0 * vol_curva / dt + q_curva
    if np.any(np.diff(func_armazenamento) <= 0):
        raise ValueError('Relação armazenamento-indicação não é crescente.')
    for k in range(1, len(entradas)):
        alvo = entradas[k-1] + entradas[k] + 2.0*s[-1]/dt - o[-1]
        if not func_armazenamento[0] <= alvo <= func_armazenamento[-1]:
            raise ValueError(f'Passo {k}: armazenamento fora das curvas. '
                             'Verifique cotas, curva de descarga e passo temporal.')
        z2 = np.interp(alvo, func_armazenamento, cotas_comuns)
        s2 = np.interp(z2, cotas_comuns, vol_curva)
        o2 = np.interp(z2, cotas_comuns, q_curva)
        z.append(z2); s.append(s2); o.append(o2)
    return pd.DataFrame({'dia': np.arange(len(entradas)), 'Q_afluente': entradas,
                         'Q_defluente': o, 'cota_m': z,
                         'volume_hm3': np.asarray(s)/1e6})

# %% 20 | Aplicar Puls com o nível inicial escolhido
resultado_puls = puls_modificado(q_afluente, q_inicial, COTA_INICIAL_M, DT_HORAS)
print(resultado_puls.round(2).to_string(index=False))
resultado_puls.to_csv(SAIDA/'transito_puls_7dias.csv', index=False)

# %% 21 | FIGURA FINAL: amortecimento da cheia e evolução do reservatório
fig, axs = plt.subplots(3, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
axs[0].plot(resultado_puls.dia, resultado_puls.Q_afluente, 'o-', color='#D55E00',
            label='Afluência prevista (LSTM Q+P)')
axs[0].plot(resultado_puls.dia, resultado_puls.Q_defluente, 's-', color='#0072B2',
            label='Defluência simulada (Puls)')
axs[0].set(ylabel='Vazão (m³/s)', title='Trânsito de cheias — sete LSTM diretas + Puls')
axs[0].legend()
axs[1].plot(resultado_puls.dia, resultado_puls.cota_m, 'o-', color='#009E73')
axs[1].axhline(COTA_INICIAL_M, ls='--', color='gray', lw=1, label='Nível inicial')
axs[1].set(ylabel='Cota do reservatório (m)')
axs[1].legend()
axs[2].plot(resultado_puls.dia, resultado_puls.volume_hm3, 'o-', color='#CC79A7')
axs[2].set(xlabel='Dias a partir da origem', ylabel='Armazenamento (hm³)')
for ax in axs:
    ax.grid(alpha=0.2)
    ax.set_xticks(np.arange(HORIZONTES + 1))
plt.show()

print('\nOBSERVAÇÃO: esta simulação usa vazões médias diárias, queda/descarga '
      'controladas pela curva e hipótese de descarga unívoca em função da cota. '
      'Não representa regras operativas, abertura de comportas ou previsão de inundação.')
print('Resultados em:', SAIDA)
print('Cenário Q+P: somente observações até a origem, sem chuva futura.')
