import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from textwrap import dedent

# Configuração do página
st.set_page_config(
    page_title="Sistema de Recomendação para Delegacias", 
    layout="wide",
    page_icon="🚔"
)

# Helpers 
@st.cache_data
def load_data():
    """
    Carrega o dataset específico do diretório Supervisionado
    """
    try:
        # Caminho específico para o dataset
        df = pd.read_csv("nao_supervisionado/dataset_delegacias.csv")
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None

@st.cache_resource
def train_recommendation_model(df):
    """
    Treina um modelo para recomendar delegacias baseado no tipo de crime
    """
    # Preparar features e target
    features = ['tipo_crime', 'mes', 'dia_semana_name']
    target = 'orgao_responsavel'
    
    # Codificar variáveis categóricas
    le_dict = {}
    X_encoded = df[features].copy()
    
    for col in features:
        if X_encoded[col].dtype == 'object':
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            le_dict[col] = le
    
    y = df[target]
    
    # Split dos dados
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Treinar modelo
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        max_depth=10,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Avaliar
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, le_dict, accuracy, features

def recommend_delegacy(tipo_crime, mes, dia_semana, model, le_dict, features):
    """
    Recomenda a delegacia baseado nas features fornecidas
    """
    try:
        # Preparar input
        input_data = {
            'tipo_crime': tipo_crime,
            'mes': mes,
            'dia_semana_name': dia_semana
        }
        
        # Codificar input
        input_encoded = []
        for feature in features:
            if feature in le_dict:
                le = le_dict[feature]
                try:
                    # Tenta transformar, se não existir usa 0 (unknown)
                    encoded_val = le.transform([input_data[feature]])[0]
                except ValueError:
                    encoded_val = 0
            else:
                encoded_val = input_data[feature]
            input_encoded.append(encoded_val)
        
        # Fazer predição
        prediction = model.predict([input_encoded])[0]
        probabilities = model.predict_proba([input_encoded])[0]
        
        # Pegar as top 3 delegacias recomendadas
        classes = model.classes_
        top_3_idx = np.argsort(probabilities)[-3:][::-1]
        top_3_delegacias = classes[top_3_idx]
        top_3_probs = probabilities[top_3_idx]
        
        return prediction, list(zip(top_3_delegacias, top_3_probs))
        
    except Exception as e:
        return None, []

# Load Data
df = load_data()

if df is None:
    st.stop()

# Sidebar - Sistema de Recomendação
st.sidebar.title("🎯 Sistema de Recomendação")
st.sidebar.markdown("---")

# Treinar modelo de recomendação
with st.sidebar:
    with st.spinner("Treinando modelo de recomendação..."):
        model, le_dict, accuracy, features = train_recommendation_model(df)

st.sidebar.success(f"Modelo treinado (Acurácia: {accuracy:.1%})")

# Inputs para recomendação
st.sidebar.subheader("Fazer Recomendação")

tipo_crime = st.sidebar.selectbox(
    "Tipo de Crime:",
    options=sorted(df['tipo_crime'].unique())
)

mes = st.sidebar.selectbox(
    "Mês:",
    options=["Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho", "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"],
)

dia_semana = st.sidebar.selectbox(
    "Dia da Semana:",
    options=["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
)

# Fazer recomendação
if st.sidebar.button("🎯 Recomendar Delegacia", type="primary"):
    delegacia_recomendada, top_3 = recommend_delegacy(tipo_crime, mes, dia_semana, model, le_dict, features)
    
    if delegacia_recomendada:
        st.sidebar.success(f"**Delegacia Recomendada:** {delegacia_recomendada}")
        
        st.sidebar.markdown("**Top 3 Recomendações:**")
        for i, (deleg, prob) in enumerate(top_3, 1):
            st.sidebar.write(f"{i}. {deleg} ({prob:.1%})")

# Sidebar - Filtros para Análise
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filtros para Análise")

# Filtros básicos
delegacias_all = sorted(df["orgao_responsavel"].dropna().unique().tolist())
crimes_all = sorted(df["tipo_crime"].dropna().unique().tolist())

sel_delegacias = st.sidebar.multiselect(
    "Delegacias:",
    options=delegacias_all,
    default=delegacias_all[:3] if len(delegacias_all) > 3 else delegacias_all
)

sel_crimes = st.sidebar.multiselect(
    "Tipos de Crime:",
    options=crimes_all,
    default=crimes_all[:3] if len(crimes_all) > 3 else crimes_all
)

# Filtro de período histórico
if 'ano' in df.columns:
    hist_year_min = int(df["ano"].min())
    hist_year_max = int(df["ano"].max())
    sel_hist_range = st.sidebar.slider(
        "Período Histórico:",
        min_value=hist_year_min,
        max_value=hist_year_max,
        value=(hist_year_min, hist_year_max)
    )

# Header Principal
st.title("🚔 Sistema de Recomendação para Delegacias")
st.markdown("**Análise inteligente e recomendações para otimização do atendimento policial**")
st.markdown("---")

# Métricas de Performance 
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_ocorrencias = df['quantidade_ocorrencia'].sum()
    st.metric("Total de Ocorrências", f"{total_ocorrencias:,}")

with col2:
    delegacias_unicas = df['orgao_responsavel'].nunique()
    st.metric("Delegacias no Sistema", delegacias_unicas)

with col3:
    crimes_unicos = df['tipo_crime'].nunique()
    st.metric("Tipos de Crime", crimes_unicos)

with col4:
    st.metric("Acurácia do Modelo", f"{accuracy:.1%}")

# Análises de Recomendação
st.subheader("📊 Análises para Otimização de Delegacias")

# Filtrar dados baseado nas seleções
df_filtrado = df.copy()
df_filtrado = df_filtrado[df_filtrado["orgao_responsavel"].isin(sel_delegacias)]
df_filtrado = df_filtrado[df_filtrado["tipo_crime"].isin(sel_crimes)]

if 'ano' in df.columns:
    df_filtrado = df_filtrado[
        (df_filtrado["ano"] >= sel_hist_range[0]) & 
        (df_filtrado["ano"] <= sel_hist_range[1])
    ]

# Layout em abas
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Matriz de Recomendação", 
    "📈 Desempenho por Delegacia", 
    "🔍 Análise Detalhada",
    "📋 Dados e Exportação"
])

with tab1:
    st.subheader("Matriz Crime vs Delegacia")
    
    # Heatmap de crimes por delegacia
    heat_data = df_filtrado.groupby(['tipo_crime', 'orgao_responsavel']).agg({
        'quantidade_ocorrencia': 'sum'
    }).reset_index()
    
    pivot_data = heat_data.pivot(
        index='tipo_crime', 
        columns='orgao_responsavel', 
        values='quantidade_ocorrencia'
    ).fillna(0)
    
    fig_heat = px.imshow(
        pivot_data,
        aspect="auto",
        title="🔥 Heatmap - Distribuição de Crimes por Delegacia",
        color_continuous_scale='Blues',
        labels={'color': 'Ocorrências'}
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    
    # Top crimes por delegacia
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Top Crimes por Delegacia**")
        top_crimes_deleg = df_filtrado.groupby(['orgao_responsavel', 'tipo_crime']).agg({
            'quantidade_ocorrencia': 'sum'
        }).reset_index()
        
        # Pegar top 3 crimes por delegacia
        top_3_per_deleg = top_crimes_deleg.groupby('orgao_responsavel').apply(
            lambda x: x.nlargest(3, 'quantidade_ocorrencia')
        ).reset_index(drop=True)
        
        for delegacia in sel_delegacias[:5]:  # Mostrar até 5 delegacias
            deleg_data = top_3_per_deleg[top_3_per_deleg['orgao_responsavel'] == delegacia]
            if not deleg_data.empty:
                st.write(f"**{delegacia}:**")
                for _, row in deleg_data.iterrows():
                    st.write(f"  - {row['tipo_crime']}: {row['quantidade_ocorrencia']} ocorrências")

with tab2:
    st.subheader("Desempenho e Eficiência")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Volume por delegacia
        vol_deleg = df_filtrado.groupby('orgao_responsavel').agg({
            'quantidade_ocorrencia': 'sum'
        }).sort_values('quantidade_ocorrencia', ascending=False)
        
        fig_vol = px.bar(
            vol_deleg, 
            x=vol_deleg.index,
            y='quantidade_ocorrencia',
            title="📊 Volume de Ocorrências por Delegacia",
            labels={'quantidade_ocorrencia': 'Total de Ocorrências', 'orgao_responsavel': 'Delegacia'}
        )
        st.plotly_chart(fig_vol, use_container_width=True)
    
    with col2:
        # Diversidade de crimes por delegacia
        diversidade = df_filtrado.groupby('orgao_responsavel').agg({
            'tipo_crime': 'nunique',
            'quantidade_ocorrencia': 'sum'
        }).reset_index()
        
        fig_div = px.scatter(
            diversidade,
            x='tipo_crime',
            y='quantidade_ocorrencia',
            size='quantidade_ocorrencia',
            color='orgao_responsavel',
            title="🎯 Diversidade vs Volume por Delegacia",
            labels={
                'tipo_crime': 'Tipos de Crime Atendidos',
                'quantidade_ocorrencia': 'Volume Total'
            }
        )
        st.plotly_chart(fig_div, use_container_width=True)

with tab3:
    st.subheader("Análise Temporal e Padrões")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Evolução temporal por delegacia
        if 'ano' in df_filtrado.columns and 'mes' in df_filtrado.columns:
            evol_temp = df_filtrado.groupby(['ano', 'mes', 'orgao_responsavel']).agg({
                'quantidade_ocorrencia': 'sum'
            }).reset_index()
            
            evol_temp['ano_mes'] = evol_temp['ano'].astype(str) + '-' + evol_temp['mes'].astype(str).str.zfill(2)
            
            fig_evol = px.line(
                evol_temp,
                x='ano_mes',
                y='quantidade_ocorrencia',
                color='orgao_responsavel',
                title="📈 Evolução Temporal por Delegacia",
                labels={'quantidade_ocorrencia': 'Ocorrências'}
            )
            st.plotly_chart(fig_evol, use_container_width=True)
    
    with col2:
         # Padrão semanal por delegacia
        if 'dia_semana_name' in df_filtrado.columns:
            dias_ordem = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dias_portugues = {
                'Monday': 'Segunda', 'Tuesday': 'Terça', 'Wednesday': 'Quarta', 
                'Thursday': 'Quinta', 'Friday': 'Sexta', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
            }
            
            padrão_semanal = df_filtrado.groupby(['dia_semana_name', 'orgao_responsavel']).agg({
                'quantidade_ocorrencia': 'sum'
            }).reset_index()
            
            padrão_semanal['dia_portugues'] = padrão_semanal['dia_semana_name'].map(dias_portugues)
            ordem_portugues = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
            padrão_semanal['dia_ordenado'] = pd.Categorical(
                padrão_semanal['dia_portugues'], 
                categories=ordem_portugues, 
                ordered=True
            )
            padrão_semanal = padrão_semanal.sort_values('dia_ordenado')
            
            fig_semana = px.line(
                padrão_semanal,
                x='dia_portugues',
                y='quantidade_ocorrencia',
                color='orgao_responsavel',
                title="📅 Padrão Semanal por Delegacia",
                labels={'quantidade_ocorrencia': 'Ocorrências'}
            )
            st.plotly_chart(fig_semana, use_container_width=True)


with tab4:
    st.subheader("Dados Detalhados e Exportação")
    
    # Estatísticas descritivas
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Estatísticas por Delegacia**")
        stats_deleg = df_filtrado.groupby('orgao_responsavel').agg({
            'quantidade_ocorrencia': ['sum', 'mean', 'count'],
            'tipo_crime': 'nunique'
        }).round(2)
        stats_deleg.columns = ['Total', 'Média', 'Qtd Registros', 'Tipos Crime Únicos']
        st.dataframe(stats_deleg)
    
    with col2:
        st.markdown("**Estatísticas por Tipo de Crime**")
        stats_crime = df_filtrado.groupby('tipo_crime').agg({
            'quantidade_ocorrencia': ['sum', 'mean', 'count'],
            'orgao_responsavel': 'nunique'
        }).round(2)
        stats_crime.columns = ['Total', 'Média', 'Qtd Registros', 'Delegacias Únicas']
        st.dataframe(stats_crime)
    
    # Dados filtrados
    st.markdown("**Dados Filtrados**")
    st.dataframe(df_filtrado.head(100), use_container_width=True)
    
    # Download
    csv = df_filtrado.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Baixar Dados Filtrados (CSV)",
        data=csv,
        file_name="dados_delegacias_filtrados.csv",
        mime="text/csv"
    )

# Footer 
st.markdown("---")
st.markdown(
    """
    **Sistema desenvolvido para otimização do atendimento policial** • 
    Use as recomendações para direcionar ocorrências às delegacias mais adequadas
    """
)