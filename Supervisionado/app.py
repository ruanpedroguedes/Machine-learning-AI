# app.py
"""
Streamlit app para visualização histórica e previsões
(quantidade de ocorrências por ano / mês / dia da semana / delegacia / tipo de crime)
Base: dataset_encoded.csv (features one-hot) + dataset_delegacias.txt (nomes originais)
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
from functools import lru_cache

st.set_page_config(page_title="Gestão de Delegacias — Painel", layout="wide")

# ---------- Helpers ----------
@st.cache_data
def load_data():
    df_enc = pd.read_csv("dataset_encoded.csv")
    df_raw = pd.read_csv("dataset_delegacias")
    # normalize column names
    return df_enc, df_raw

@st.cache_resource
def train_model(df_enc):
    X = df_enc.drop(columns=["quantidade_ocorrencia", "tipo_crime"])
    y = df_enc["quantidade_ocorrencia"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5   # cálculo manual do RMSE (sem usar squared=False)
    r2 = r2_score(y_test, y_pred)

    return model, {"mae": mae, "rmse": rmse, "r2": r2}, list(X.columns)


def build_feature_row_for_combo(col_names, delegacia, dia_semana, tipo_crime, ano, mes):
    # create a zero row dict matching training columns
    row = {c: 0 for c in col_names}
    # set ano and mes
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

# show metrics
st.header("Painel de Gestão de Ocorrências — Delegacias")
col1, col2, col3 = st.columns([1,1,2])
with col1:
    st.metric("MAE (teste)", f"{metrics['mae']:.3f}")
with col2:
    st.metric("RMSE (teste)", f"{metrics['rmse']:.3f}")
with col3:
    st.metric("R² (teste)", f"{metrics['r2']:.3f}")

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
if not future_X.empty:
    preds = model.predict(future_X)
    df_future_preds = future_X.copy()
    df_future_preds["predicted"] = preds
    # We need also columns to group by readable keys (ano, mes, delegacia, dia, crime)
    # But training features are one-hot. We'll extract the readable labels by reverse mapping:
    # extract delegacia and crime and dia from one-hot columns
    # helper functions:
    def find_onehot_label(row, prefix):
        for c,v in row.items():
            if c.startswith(prefix) and (v == 1 or v==1.0):
                return c.replace(prefix, "")
        return None

    # build nice columns
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
    st.dataframe(hist_grp.head(30))
with right_col:
    st.subheader("Previsões")
    if fig_pred is not None:
        st.plotly_chart(fig_pred, use_container_width=True)
        st.markdown("**Tabela de previsões (amostra)**")
        st.dataframe(pred_grp.head(30))
        # Download CSV
        csv = pred_grp.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Baixar previsões (CSV)", data=csv, file_name="previsoes_agregadas.csv", mime="text/csv")
    else:
        st.info("Sem previsões (combinações vazias). Ajuste filtros/seleções no sidebar.")

st.markdown("---")
st.caption("Observação: previsões geradas pelo RandomForest treinado com as features one-hot presentes em dataset_encoded.csv. Agregamos as combinações (delegacia × crime × mês × dia) por ano/mês/dia/delegacia/tipo para obter os totais esperados.")
