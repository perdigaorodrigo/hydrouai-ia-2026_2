"""Aula prática: coleta de notícias de alagamentos em BH, em células do Spyder.

Instalação: pip install pandas requests feedparser python-dateutil tqdm
Execução: python -u 01_coletar_google_news_bh.py
"""
import json
import random
import re
import time
import unicodedata
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import feedparser
import pandas as pd
import requests
from dateutil import parser as dateparser
from tqdm import tqdm

# %% 1 | Bibliotecas e configurações
# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================
PASTA = Path(__file__).resolve().parent
ANOS = [2019, 2020, 2021, 2022]
JANELA_DIAS = 30

TERMOS = [
    "alagamento",
    "alagamentos",
    "vias alagadas",
    "chuva alaga",
    "chuva alagou",
]

# False = exige a expressão "Belo Horizonte"; True = também aceita "BH".
# Incluir BH aumenta a cobertura, mas pode aumentar os falsos positivos.
INCLUIR_SIGLA_BH = False  # Não é utilizado na consulta restrita por intitle

PAUSA_MIN = 3
PAUSA_MAX = 6
MAX_TENTATIVAS = 3
TIMEOUT = 30
LIMITE_ALERTA_RSS = 95

ARQUIVO_BRUTO = PASTA / "google_news_bh_bruto.csv"
ARQUIVO_UNICO = PASTA / "google_news_bh_unico.csv"
ARQUIVO_CHECKPOINT = PASTA / "checkpoint_google_news_bh.json"
ARQUIVO_FALHAS = PASTA / "google_news_bh_falhas.csv"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140.0 Safari/537.36"
)
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "pt-BR,pt;q=0.9"})

if not ANOS or any(not isinstance(a, int) or not 1900 <= a <= 9998 for a in ANOS):
    raise ValueError("ANOS deve conter anos inteiros entre 1900 e 9998.")
if JANELA_DIAS < 1:
    raise ValueError("JANELA_DIAS deve ser positivo.")
ANOS = sorted(set(ANOS))

# %% 2 | Funções auxiliares: normalização, datas e janelas
# ============================================================
# 2. FUNÇÕES AUXILIARES
# ============================================================
def normalizar(texto):
    texto = unicodedata.normalize("NFKD", str(texto or "").lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", texto).strip()


def carregar_json(caminho, padrao):
    if not caminho.exists():
        return padrao
    with caminho.open("r", encoding="utf-8") as f:
        return json.load(f)


def salvar_json(caminho, dados):
    temporario = caminho.with_suffix(".tmp")
    with temporario.open("w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)
    temporario.replace(caminho)


def gerar_janelas(inicio, fim):
    atual = pd.Timestamp(inicio)
    limite = pd.Timestamp(fim)  # limite exclusivo
    while atual < limite:
        proximo = min(atual + pd.Timedelta(days=JANELA_DIAS), limite)
        yield atual.date(), proximo.date()
        atual = proximo


def montar_query(termo, inicio, fim):
    return (
        f'intitle:"Belo Horizonte" "{termo}" '
        f'after:{inicio.isoformat()} '
        f'before:{fim.isoformat()}'
    )


def montar_url_rss(query):
    parametros = {"q": query, "hl": "pt-BR", "gl": "BR", "ceid": "BR:pt-419"}
    return "https://news.google.com/rss/search?" + urlencode(parametros)


def converter_data(valor):
    if not valor:
        return None
    try:
        data = dateparser.parse(str(valor))
        if data.tzinfo is None:
            data = data.replace(tzinfo=timezone.utc)
        return data.isoformat()
    except (TypeError, ValueError, OverflowError):
        return None


def separar_titulo_fonte(entry):
    titulo_original = entry.get("title", "").strip()
    fonte_obj = entry.get("source", {})
    fonte = fonte_obj.get("title", "") if isinstance(fonte_obj, dict) else ""
    titulo = titulo_original
    if fonte and titulo.endswith(" - " + fonte):
        titulo = titulo[: -(len(fonte) + 3)].strip()
    return titulo, fonte


def chave_noticia(row):
    return normalizar(row["titulo"]) + "|" + normalizar(row["fonte"])

# %% 3 | Consulta ao Google News RSS
# ============================================================
# 3. CONSULTAR GOOGLE NEWS
# ============================================================
def consultar_google_news(query):
    url = montar_url_rss(query)
    ultimo_erro = None
    for tentativa in range(1, MAX_TENTATIVAS + 1):
        try:
            resposta = SESSION.get(url, timeout=TIMEOUT)
            resposta.raise_for_status()
            feed = feedparser.parse(resposta.content)
            if feed.bozo and not feed.entries:
                raise RuntimeError(f"RSS inválido: {feed.bozo_exception}")
            if not feed.entries and (
                "xml" not in resposta.headers.get("Content-Type", "").lower()
                and "rss" not in resposta.text[:500].lower()
            ):
                raise RuntimeError("Resposta não parece ser um feed RSS")
            return feed.entries, url
        except Exception as exc:
            ultimo_erro = str(exc)
            if tentativa < MAX_TENTATIVAS:
                espera = min(60, 10 * tentativa)
                print(f"\nFalha na tentativa {tentativa}: {ultimo_erro}")
                time.sleep(espera)
    raise RuntimeError(ultimo_erro)

# %% 4 | Extrair e filtrar os metadados das notícias
# ============================================================
# 4. EXTRAIR METADADOS
# ============================================================
def extrair_registros(entries, termo, query, inicio, fim, url_rss):
    registros = []
    for entry in entries:
        titulo, fonte = separar_titulo_fonte(entry)
        titulo_normalizado = normalizar(titulo)
        if "belo horizonte" not in titulo_normalizado:
            continue
        registros.append({
            "localidade_busca": "Belo Horizonte",
            "uf_busca": "MG",
            "fenomeno_busca": termo,
            "inicio_consulta": inicio.isoformat(),
            "fim_consulta_exclusivo": fim.isoformat(),
            "query": query,
            "titulo": titulo,
            "fonte": fonte,
            "data_publicacao": converter_data(entry.get("published", "")),
            "url": entry.get("link", ""),
            "google_news_id": entry.get("id", ""),
            "url_rss": url_rss,
            "titulo_menciona_bh": (
                "belo horizonte" in titulo_normalizado
                or "bh" in titulo_normalizado.split()
            ),
            "coleta_em": datetime.now(timezone.utc).isoformat(),
        })
    return registros

# %% 5 | Organizar, deduplicar e salvar os CSVs
# ============================================================
# 5. SALVAR RESULTADOS
# ============================================================
COLUNAS = [
    "localidade_busca", "uf_busca", "fenomeno_busca", "inicio_consulta",
    "fim_consulta_exclusivo", "query", "titulo", "fonte", "data_publicacao",
    "url", "google_news_id", "url_rss", "titulo_menciona_bh", "coleta_em",
]


def salvar_bases(registros):
    df = pd.DataFrame(registros, columns=COLUNAS)
    df.to_csv(ARQUIVO_BRUTO, sep=";", index=False, encoding="utf-8-sig")
    if df.empty:
        df.to_csv(ARQUIVO_UNICO, sep=";", index=False, encoding="utf-8-sig")
        return df, df
    df["chave_unica"] = df.apply(chave_noticia, axis=1)
    df_unico = df.drop_duplicates(subset=["chave_unica"]).copy()
    df_unico.drop(columns=["chave_unica"], inplace=True)
    df_unico.sort_values("data_publicacao", na_position="last", inplace=True)
    df_unico.to_csv(ARQUIVO_UNICO, sep=";", index=False, encoding="utf-8-sig")
    return df, df_unico

# ============================================================
# 6. EXECUÇÃO PRINCIPAL
# ============================================================
# %% 6 | Planejar as buscas (exemplo: janeiro de 2020)

# Cada termo é consultado em janelas de 30 dias.
janelas = []
for ano in ANOS:
    janelas.extend(gerar_janelas(date(ano, 1, 1), date(ano + 1, 1, 1)))

consultas = [
    {"termo": termo, "inicio": inicio, "fim": fim,
     "query": montar_query(termo, inicio, fim)}
    for termo in TERMOS
    for inicio, fim in janelas
]

print(f"Anos: {ANOS} | Termos: {len(TERMOS)} | Consultas: {len(consultas)}")

# Exemplo didático: janeiro de 2020
termo_exemplo = "alagamentos"
inicio_exemplo = date(2020, 1, 1)
fim_exemplo = date(2020, 2, 1)

query_exemplo = montar_query(
    termo_exemplo,
    inicio_exemplo,
    fim_exemplo
)

print("Exemplo de consulta:", query_exemplo)
print("Exemplo de URL:", montar_url_rss(query_exemplo))


# %% 7 | Testar notícia conhecida de janeiro de 2020

query = (
    '"alagamentos" '
    'after:2020-01-01 before:2020-02-01'
)

entradas, url = consultar_google_news(query)

print("Notícias encontradas:", len(entradas))
print("URL:", url)

for noticia in entradas:
    print(noticia.get("title"))

# %% 8 | Carregar checkpoint para retomar coletas anteriores
checkpoint = carregar_json(ARQUIVO_CHECKPOINT, {
    "consultas_concluidas": [], "registros": [], "falhas": []
})
concluidas = set(checkpoint["consultas_concluidas"])
registros = checkpoint["registros"]
falhas = checkpoint["falhas"]
print(f"Consultas concluídas anteriormente: {len(concluidas)}")


# %% 9 | Executar a coleta (esta célula pode levar vários minutos)
# A cada consulta, o progresso é salvo no checkpoint.
try:
    for consulta in tqdm(consultas, desc="Consultas Google News"):
        query = consulta["query"]
        if query in concluidas:
            continue
        try:
            entries, url_rss = consultar_google_news(query)
            novos = extrair_registros(
                entries, consulta["termo"], query,
                consulta["inicio"], consulta["fim"], url_rss
            )
            registros.extend(novos)
            print(f"\n{consulta['termo']} | {consulta['inicio']} | "
                  f"{len(entries)} entradas, {len(novos)} retidas")
            if len(entries) >= LIMITE_ALERTA_RSS:
                print("ATENÇÃO: RSS próximo do limite; considere reduzir JANELA_DIAS.")
            concluidas.add(query)
            falhas = [f for f in falhas if f.get("query") != query]
            checkpoint.update({
                "consultas_concluidas": sorted(concluidas),
                "registros": registros, "falhas": falhas
            })
            salvar_json(ARQUIVO_CHECKPOINT, checkpoint)
        except Exception as exc:
            print(f"\nERRO: {query}\n{exc}")
            falhas = [f for f in falhas if f.get("query") != query]
            falhas.append({"query": query, "erro": str(exc)})
            checkpoint["falhas"] = falhas
            salvar_json(ARQUIVO_CHECKPOINT, checkpoint)
        time.sleep(random.uniform(PAUSA_MIN, PAUSA_MAX))
except KeyboardInterrupt:
    print("Coleta interrompida. Progresso preservado no checkpoint.")


# %% 10 | Exportar notícias e relatório de falhas
# Inclui apenas consultas da configuração atual, mesmo com checkpoint antigo.
queries_atuais = {c["query"] for c in consultas}
registros_selecionados = [
    r for r in registros if r.get("query") in queries_atuais
]
df_bruto, df_unico = salvar_bases(registros_selecionados)
falhas_atuais = [f for f in falhas if f.get("query") in queries_atuais]
pd.DataFrame(falhas_atuais, columns=["query", "erro"]).to_csv(
    ARQUIVO_FALHAS, sep=";", index=False, encoding="utf-8-sig"
)
print(f"Notícias brutas: {len(df_bruto)}")
print(f"Notícias únicas: {len(df_unico)}")
print(f"Consultas com falha: {len(falhas_atuais)}")


# %% 11 | Explorar os resultados no Spyder
# Notícias não são eventos únicos: a validação ocorrerá na etapa do Ollama.
if not df_unico.empty:
    df_unico["ano_publicacao"] = pd.to_datetime(
        df_unico["data_publicacao"], errors="coerce", utc=True
    ).dt.year
    print("\nNotícias por ano:")
    print(df_unico["ano_publicacao"].value_counts().sort_index())
    print("\nNotícias por termo:")
    print(df_unico["fenomeno_busca"].value_counts())
    print("\nExemplos:")
    print(df_unico[["titulo", "fonte", "data_publicacao"]].head(10))
else:
    print("Nenhuma notícia coletada nas consultas selecionadas.")

print("\nArquivos:", ARQUIVO_BRUTO, ARQUIVO_UNICO, ARQUIVO_FALHAS, sep="\n")
