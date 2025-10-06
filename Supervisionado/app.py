# app.py (versão atualizada)
"""
Streamlit app para visualização histórica e previsões
(quantidade de ocorrências por ano / mês / dia da semana / delegacia / tipo de crime)
Melhorias:
- organização dos gráficos de previsões quando há apenas 1 ano previsto (grid de pequenos gráficos)
- explicação didática das métricas (pensado para policial/leigo)
- novos gráficos de análise exploratória
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
from functools import lru_cache
import math
from textwrap import dedent

st.set_page_config(page_title="Gestão de Delegacias — Painel", layout="wide")

# ---------- Helpers ----------
@st.cache_data
def load_data():
    """
    Tenta carregar os datasets mais comuns que você pode ter enviado.
    Retorna: df_enc (one-hot), df_raw (original)
    """
    enc_paths = ["dataset_encoded.csv", "dataset_encoded (1).csv", "dataset_encoded.csv"]
    raw_paths = ["dataset_delegacias.txt", "dataset_delegacias (1).txt", "dataset_delegacias.csv", "dataset_delegacias"]

    df_enc = None
    df_raw = None
    # tenta arquivos possíveis para dataset encoded
    for p in enc_paths:
        try:
            df_enc = pd.read_csv(p)
            break
        except Exception:
            continue
    # tenta arquivos possíveis para raw
    for p in raw_paths:
        try:
            # se for txt com separador por vírgula, pandas lê normalmente
            df_raw = pd.read_csv(p)
            break
        except Exception:
            continue

    if df_enc is None or df_raw is None:
        raise FileNotFoundError("Não foi possível localizar 'dataset_encoded' ou 'dataset_delegacias'. Verifique os nomes dos arquivos no diretório.")

    # Normaliza nomes de colunas textos consistentes
    df_raw.columns = [c.strip() for c in df_raw.columns]
    return df_enc, df_raw

@st.cache_resource
def train_model(df_enc):
    """
    Treina um RandomForest simples e retorna o modelo, métricas e lista de features usadas.
    """
    # evita colunas inesperadas
    X = df_enc.drop(columns=[c for c in ["quantidade_ocorrencia", "tipo_crime"] if c in df_enc.columns], errors="ignore")
    y = df_enc["quantidade_ocorrencia"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return model, {"mae": mae, "rmse": rmse, "r2": r2}, list(X.columns)

def build_feature_row_for_combo(col_names, delegacia, dia_semana, tipo_crime, ano, mes):
    # create a zero row dict matching training columns
    row = {c: 0 for c in col_names}
    # set ano and mes if present among features
    if "ano" in row: row["ano"] = int(ano)
    if "mes" in row: row["mes"] = int(mes)
    # set delegacia one-hot if present
    org_col = f"orgao_responsavel_{delegacia}"
    if org_col in row:
        row[org_col] = 1
    # set dia da semana
    dia_col = f"dia_semana_name_{dia_semana}"
    if dia_col in row:
        row[dia_col] = 1
    # set crime type
    crime_col = f"tipo_crime_{tipo_crime}"
    if crime_col in row:
        row[crime_col] = 1
    return row

def generate_future_df(col_names, delegacias, crimes, years, months, dias):
    rows = []
    for ano in years:
        for mes in months:
            for deleg in delegacias:
                for crime in crimes:
                    for dia in dias:
                        rows.append(build_feature_row_for_combo(col_names, deleg, dia, crime, ano, mes))
    if not rows:
        return pd.DataFrame(columns=col_names)
    df_future = pd.DataFrame(rows)
    # Ensure column order
    df_future = df_future.reindex(columns=col_names, fill_value=0)
    return df_future

def chunk_dataframe(df, chunk_size):
    """Divide um dataframe em pedaços para exibir em múltiplos gráficos."""
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i+chunk_size]

def top_n_and_rest(df, key_col, value_col, n=8):
    """Retorna top n e o restante agregado como 'Outros'"""
    grp = df.groupby(key_col, as_index=False).agg({value_col:"sum"}).sort_values(value_col, ascending=False)
    if len(grp) <= n:
        return grp
    top = grp.head(n)
    rest = grp.tail(len(grp)-n)
    rest_sum = rest[value_col].sum()
    top = pd.concat([top, pd.DataFrame([{key_col: "Outros", value_col: rest_sum}])], ignore_index=True)
    return top

# ---------- Load ----------
df_enc, df_raw = load_data()

# Unique values for controls (from raw data for readable names)
delegacias_all = sorted(df_raw["orgao_responsavel"].dropna().unique().tolist())
crimes_all = sorted(df_raw["tipo_crime"].dropna().unique().tolist())
dias_all = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

# ---------- Sidebar controls ----------
st.sidebar.title("Filtros & Previsões")
sel_delegacias = st.sidebar.multiselect("Delegacias (selecione)", options=delegacias_all, default=delegacias_all)
sel_crimes = st.sidebar.multiselect("Tipos de crime (filtro avançado)", options=crimes_all, default=crimes_all)
agg_choice = st.sidebar.radio("Agregação principal (visualização):", ["Ano", "Mês", "Dia da semana", "Delegacia", "Tipo de crime"])
years_to_forecast = st.sidebar.number_input("Prever para quantos anos à frente?", min_value=1, max_value=5, value=1)
months_to_include = st.sidebar.multiselect("Meses a considerar nas previsões (padrão: todos)", options=list(range(1,13)), default=list(range(1,13)))
dias_selected = st.sidebar.multiselect("Dias da semana a considerar (previsões)", options=dias_all, default=dias_all)
hist_year_min = int(df_raw["ano"].min())
hist_year_max = int(df_raw["ano"].max())
sel_hist_range = st.sidebar.slider("Período histórico (anos) exibido", min_value=hist_year_min, max_value=hist_year_max, value=(hist_year_min, hist_year_max))

st.sidebar.markdown("---")
st.sidebar.write("Modelo: RandomForestRegressor (re-treinado localmente ao carregar)")
st.sidebar.info("Dica: quanto mais anos/combinações pedir no forecast, maior o tempo de execução — o app treina rapidamente neste dataset.")

# ---------- Train model ----------
with st.spinner("Treinando modelo (rápido) ..."):
    model, metrics, feature_cols = train_model(df_enc)

# ---------- Header + Métricas didáticas ----------
st.header("Painel de Gestão de Ocorrências — Delegacias")

st.markdown(dedent("""
**Como interpretar os indicadores do modelo (de forma simples):**

- 📉 **Erro médio (MAE)** — mostra, em média, quanto o modelo **erra** na previsão (por exemplo: MAE = 0.3 significa que, em média, a previsão difere 0.3 ocorrências do observado).  
- 📏 **Variação do erro (RMSE)** — similar ao MAE, mas dá mais peso a grandes erros; ajuda a entender a **variabilidade** dos erros.  
- 🎯 **Qualidade da explicação (R²)** — indica quanto das variações históricas o modelo consegue explicar; 1.0 = perfeito, 0 = não explica nada.

> Dica: esses números servem para **entender a confiança do sistema** — você não precisa modificá-los, o painel já mostra as previsões prontas para uso.
"""))

col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.metric("Erro Médio (MAE)", f"{metrics['mae']:.3f}")
with col2:
    st.metric("Erro Quadrático (RMSE)", f"{metrics['rmse']:.3f}")
with col3:
    st.metric("Qualidade (R²)", f"{metrics['r2']:.3f}")

with st.expander("O que isso significa na prática?"):
    st.write(dedent("""
        - Um **MAE pequeno** indica que, em média, as previsões estão próximas ao observado.  
        - Um **RMSE maior que o MAE** indica que existem alguns erros grandes (outliers) nas previsões.  
        - Um **R² próximo de 1** significa que o modelo capta bem os padrões históricos; valores intermediários (ex.: 0.3–0.6) mostram que o modelo tem utilidade porém não é perfeito.
    """))

# ---------- Historical aggregation ----------
df_hist = df_raw.copy()
# filter historical range
df_hist = df_hist[(df_hist["ano"] >= sel_hist_range[0]) & (df_hist["ano"] <= sel_hist_range[1])]
# filter by delegacia and crime
df_hist = df_hist[df_hist["orgao_responsavel"].isin(sel_delegacias)]
df_hist = df_hist[df_hist["tipo_crime"].isin(sel_crimes)]

# Aggregation mapping
if agg_choice == "Ano":
    hist_grp = df_hist.groupby("ano", as_index=False).agg({"quantidade_ocorrencia":"sum"}).sort_values("ano")
    fig_hist = px.bar(hist_grp, x="ano", y="quantidade_ocorrencia", title="Ocorrências por Ano (Histórico)", labels={"quantidade_ocorrencia":"Ocorrências"})
elif agg_choice == "Mês":
    hist_grp = df_hist.groupby(["ano","mes"], as_index=False).agg({"quantidade_ocorrencia":"sum"})
    hist_grp["ano_mes"] = hist_grp["ano"].astype(str) + "-" + hist_grp["mes"].astype(str).str.zfill(2)
    fig_hist = px.bar(hist_grp.sort_values(["ano","mes"]), x="ano_mes", y="quantidade_ocorrencia", title="Ocorrências por Mês (Histórico)", labels={"quantidade_ocorrencia":"Ocorrências"})
elif agg_choice == "Dia da semana":
    hist_grp = df_hist.groupby("dia_semana_name", as_index=False).agg({"quantidade_ocorrencia":"sum"})
    # keep order Monday...Sunday
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    hist_grp["dia_semana_name"] = pd.Categorical(hist_grp["dia_semana_name"], categories=order, ordered=True)
    hist_grp = hist_grp.sort_values("dia_semana_name")
    fig_hist = px.bar(hist_grp, x="dia_semana_name", y="quantidade_ocorrencia", title="Ocorrências por Dia da Semana (Histórico)", labels={"quantidade_ocorrencia":"Ocorrências"})
elif agg_choice == "Delegacia":
    hist_grp = df_hist.groupby("orgao_responsavel", as_index=False).agg({"quantidade_ocorrencia":"sum"}).sort_values("quantidade_ocorrencia", ascending=False)
    fig_hist = px.bar(hist_grp, x="orgao_responsavel", y="quantidade_ocorrencia", title="Ocorrências por Delegacia (Histórico)", labels={"quantidade_ocorrencia":"Ocorrências"})
else: # Tipo de crime
    hist_grp = df_hist.groupby("tipo_crime", as_index=False).agg({"quantidade_ocorrencia":"sum"}).sort_values("quantidade_ocorrencia", ascending=False)
    fig_hist = px.bar(hist_grp, x="tipo_crime", y="quantidade_ocorrencia", title="Ocorrências por Tipo de Crime (Histórico)", labels={"quantidade_ocorrencia":"Ocorrências"})

# ---------- Predictions ----------
# determine years to forecast
last_year = int(df_raw["ano"].max())
years_future = [last_year + i for i in range(1, years_to_forecast + 1)]

months_sel = months_to_include if months_to_include else list(range(1,13))
dias_sel = dias_selected if dias_selected else dias_all

# generate combos (future)
future_X = generate_future_df(feature_cols, sel_delegacias, sel_crimes, years_future, months_sel, dias_sel)

predicted = []
fig_pred = None
pred_grp = pd.DataFrame()
if not future_X.empty:
    preds = model.predict(future_X)
    df_future_preds = future_X.copy()
    df_future_preds["predicted"] = preds

    # build readable rows
    readable_rows = []
    for i, row in df_future_preds.iterrows():
        rdict = {}
        rdict["ano"] = int(row["ano"]) if "ano" in row else None
        rdict["mes"] = int(row["mes"]) if "mes" in row else None
        # find delegacia
        deleg = None
        for c in feature_cols:
            if c.startswith("orgao_responsavel_") and row.get(c,0)==1:
                deleg = c.replace("orgao_responsavel_","")
                break
        rdict["delegacia"] = deleg
        # find dia
        dia = None
        for c in feature_cols:
            if c.startswith("dia_semana_name_") and row.get(c,0)==1:
                dia = c.replace("dia_semana_name_","")
                break
        rdict["dia_semana"] = dia
        # find crime
        crime = None
        for c in feature_cols:
            if c.startswith("tipo_crime_") and row.get(c,0)==1:
                crime = c.replace("tipo_crime_","")
                break
        rdict["tipo_crime"] = crime
        rdict["predicted"] = row["predicted"]
        readable_rows.append(rdict)
    df_preds_readable = pd.DataFrame(readable_rows)

    # Aggregations for display
    if agg_choice == "Ano":
        pred_grp = df_preds_readable.groupby("ano", as_index=False).agg({"predicted":"sum"}).sort_values("ano")
        fig_pred = px.bar(pred_grp, x="ano", y="predicted", title=f"Previsões por Ano (próx. {years_to_forecast} anos)", labels={"predicted":"Ocorrências previstas"})
    elif agg_choice == "Mês":
        pred_grp = df_preds_readable.groupby(["ano","mes"], as_index=False).agg({"predicted":"sum"})
        pred_grp["ano_mes"] = pred_grp["ano"].astype(str) + "-" + pred_grp["mes"].astype(str).str.zfill(2)
        fig_pred = px.bar(pred_grp.sort_values(["ano","mes"]), x="ano_mes", y="predicted", title="Previsões por Mês", labels={"predicted":"Ocorrências previstas"})
    elif agg_choice == "Dia da semana":
        pred_grp = df_preds_readable.groupby("dia_semana", as_index=False).agg({"predicted":"sum"})
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        pred_grp["dia_semana"] = pd.Categorical(pred_grp["dia_semana"], categories=order, ordered=True)
        pred_grp = pred_grp.sort_values("dia_semana")
        fig_pred = px.bar(pred_grp, x="dia_semana", y="predicted", title="Previsões por Dia da Semana", labels={"predicted":"Ocorrências previstas"})
    elif agg_choice == "Delegacia":
        pred_grp = df_preds_readable.groupby("delegacia", as_index=False).agg({"predicted":"sum"}).sort_values("predicted", ascending=False)
        fig_pred = px.bar(pred_grp, x="delegacia", y="predicted", title="Previsões por Delegacia", labels={"predicted":"Ocorrências previstas"})
    else:
        pred_grp = df_preds_readable.groupby("tipo_crime", as_index=False).agg({"predicted":"sum"}).sort_values("predicted", ascending=False)
        fig_pred = px.bar(pred_grp, x="tipo_crime", y="predicted", title="Previsões por Tipo de Crime", labels={"predicted":"Ocorrências previstas"})
else:
    fig_pred = None

# ---------- Layout ----------
left_col, right_col = st.columns((1,1))
with left_col:
    st.subheader("Histórico")
    st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown("**Tabela histórica (amostra)**")
    try:
        st.dataframe(hist_grp.head(30))
    except Exception:
        st.write("Sem dados históricos para mostrar nesta agregação/filtragem.")

with right_col:
    st.subheader("Previsões")
    if fig_pred is not None:
        # Caso especial: quando só houver 1 ano previsto, evitar gráfico 'blocão'
        if years_to_forecast == 1:
            st.markdown("**Previsões organizadas (1 ano selecionado) — pequenos gráficos para facilitar visualização**")
            # dependendo da agregação, mostramos pequenos gráficos por grupo (tipo_crime ou delegacia)
            if agg_choice in ["Tipo de crime", "Delegacia"]:
                # escolher coluna chave
                key = "tipo_crime" if agg_choice == "Tipo de crime" else "delegacia"
                # limitar à top N (para não criar 100 gráficos)
                top_df = top_n_and_rest(pred_grp, key, "predicted", n=8)
                # cria chunks para exibir 4 por linha
                chunk_size = 4
                chunks = list(chunk_dataframe(top_df, chunk_size))
                for chunk in chunks:
                    cols = st.columns(len(chunk))
                    for ci, (_, r) in enumerate(chunk.iterrows()):
                        with cols[ci]:
                            single = pd.DataFrame([{key: r[key], "predicted": r["predicted"]}])
                            # gráfico pequeno
                            fig_small = px.bar(single, x=key, y="predicted", title=str(r[key]), labels={"predicted":"Ocorrências previstas"})
                            fig_small.update_layout(height=240, margin=dict(t=40, b=20, l=20, r=20))
                            st.plotly_chart(fig_small, use_container_width=True)
                # também mostramos tabela agregada menor abaixo
                st.markdown("**Resumo agregado (amostra)**")
                st.dataframe(top_df.head(30))
                # download
                csv = top_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Baixar previsões (CSV)", data=csv, file_name="previsoes_agregadas.csv", mime="text/csv")
            else:
                # para outras agregações, só mostramos o gráfico normal com altura reduzida
                fig_pred.update_layout(height=420, bargap=0.18)
                st.plotly_chart(fig_pred, use_container_width=True)
                st.markdown("**Tabela de previsões (amostra)**")
                try:
                    st.dataframe(pred_grp.head(30))
                    csv = pred_grp.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Baixar previsões (CSV)", data=csv, file_name="previsoes_agregadas.csv", mime="text/csv")
                except Exception:
                    st.write("Nenhuma tabela de previsões disponível.")
        else:
            # quando >1 ano, exibe gráfico normal mas com altura moderada
            fig_pred.update_layout(height=520, bargap=0.15)
            st.plotly_chart(fig_pred, use_container_width=True)
            st.markdown("**Tabela de previsões (amostra)**")
            try:
                st.dataframe(pred_grp.head(30))
                csv = pred_grp.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Baixar previsões (CSV)", data=csv, file_name="previsoes_agregadas.csv", mime="text/csv")
            except Exception:
                st.write("Nenhuma tabela de previsões disponível.")
    else:
        st.info("Sem previsões (combinações vazias). Ajuste filtros/seleções no sidebar.")



# ---------- Análises Exploratórias (nova seção) ----------
st.markdown("## 🔍 Análises Exploratórias (Insights)")

# 1) Crimes mais recorrentes (geral)
colA, colB = st.columns(2)
with colA:
    st.markdown("**Crimes mais recorrentes (total)**")
    try:
        tot_crimes = df_raw.groupby("tipo_crime", as_index=False).agg({"quantidade_ocorrencia":"sum"}).sort_values("quantidade_ocorrencia", ascending=False)
        top_crimes = top_n_and_rest(tot_crimes, "tipo_crime", "quantidade_ocorrencia", n=12)
        fig1 = px.bar(top_crimes, x="tipo_crime", y="quantidade_ocorrencia", title="Crimes mais recorrentes (total)", labels={"quantidade_ocorrencia":"Ocorrências"})
        fig1.update_layout(height=420)
        st.plotly_chart(fig1, use_container_width=True)
    except Exception as e:
        st.write("Erro ao gerar gráfico de crimes:", e)

with colB:
    st.markdown("**Distribuição por dia da semana**")
    try:
        box = px.box(df_raw, x="dia_semana_name", y="quantidade_ocorrencia", title="Distribuição de Ocorrências por Dia da Semana", labels={"quantidade_ocorrencia":"Ocorrências"})
        box.update_layout(height=420)
        st.plotly_chart(box, use_container_width=True)
    except Exception as e:
        st.write("Erro ao gerar boxplot:", e)

# 2) Heatmap ano x tipo_crime
st.markdown("**Mapa de calor — Tipo de Crime × Ano**")
try:
    heat = df_raw.groupby(["ano","tipo_crime"], as_index=False).agg({"quantidade_ocorrencia":"sum"})
    # pivot
    heat_pivot = heat.pivot(index="tipo_crime", columns="ano", values="quantidade_ocorrencia").fillna(0)
    fig2 = px.imshow(heat_pivot, aspect="auto", origin="lower", title="Mapa de Calor — Tipo de Crime por Ano")
    fig2.update_layout(height=420)
    st.plotly_chart(fig2, use_container_width=True)
except Exception as e:
    st.write("Erro ao gerar heatmap:", e)

# 3) Top delegacias por tipo de crime (sunburst ou treemap)
st.markdown("**Top delegacias para cada tipo de crime (treemap)**")
try:
    tre = df_raw.groupby(["tipo_crime","orgao_responsavel"], as_index=False).agg({"quantidade_ocorrencia":"sum"})
    top_tre = tre.sort_values("quantidade_ocorrencia", ascending=False).groupby("tipo_crime").head(8)
    fig3 = px.treemap(top_tre, path=["tipo_crime","orgao_responsavel"], values="quantidade_ocorrencia", title="Onde cada crime mais acontece (amostra)")
    fig3.update_layout(height=600)
    st.plotly_chart(fig3, use_container_width=True)
except Exception as e:
    st.write("Erro ao gerar treemap:", e)

