# Machine-learning-AI

![Status](https://img.shields.io/badge/status-em%20concluido-blue) 

Projeto de Inteligência Artificial e Aprendizado de Máquina aplicado à Polícia Civil de Pernambuco, com foco na gestão de delegacias e na entrega de informações relevantes e de valor prático para a melhoria da eficiência e da tomada de decisões dentro da instituição.

**Objetivo:** Desenvolver soluções baseadas em Machine Learning capazes de gerar insights úteis e aplicáveis à realidade da Polícia Civil, contribuindo para a otimização de processos e aprimoramento da gestão das unidades policiais.

**Integrantes do grupo:**

- Antonio Lemos
- Ruan Guedes
- Erick Carrasco
- Dandara Gouveia
- Gabriel Afonso
- João Lucas
- Jairo Marinho

**Aplicação**
- [Sistema de Otimização de Delegacia](https://machine-learning-ai-d2yfplzz2jkqtry3cxfmmt.streamlit.app/)

**Design Figma e Apresentação**
- [Figma](https://www.figma.com/design/wOeRvufftfiorEAUbSIYtu/Machine-learning-AI?node-id=0-1&t=1cYXgcJJ6imD2La5-1)
- [Apresentação](https://www.canva.com/design/DAG1H4M5ckM/SoWyjv2lFYr6my4fgZiLKQ/edit?utm_content=DAG1H4M5ckM&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)
- [Documentos de Pitch e modelo de negócios](https://www.canva.com/design/DAG096B2D4A/v6hnSsvxeB04EkotqAK4Nw/edit?utm_content=DAG096B2D4A&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

---

# Painel de Gestão — Supervisionado

App **Streamlit** para visualização histórica e previsões de ocorrências por delegacia, ano, mês, dia da semana e tipo de crime.

## Estrutura geral de pastas
```
.
├── NaoSupervisionado/
│   ├── EDA.ipynb
│   └── modelo.ipynb
│
├── Supervisionado/
│   ├── app.py
│   ├── dataset_delegacias/
│   ├── dataset_encoded.csv
│   ├── EDA.ipynb
│   └── modelo.ipynb
│
├── .gitignore
├── dataset_encoded.csv
├── dataset_ocorrencias_delegacia_5.csv
├── README.md
└── requirements.txt
```

## Estrutura do projeto

Coloque estes arquivos dentro da pasta `Supervisionado`:
```
Supervisionado/
├─ app.py
├─ dataset_encoded.csv
├─ dataset_delegacias.txt
├─ requirements.txt
```
yaml
Copiar código

### Resumo técnico dos arquivos

- **dataset_delegacias.txt / CSV original:** contém colunas  
  `orgao_responsavel, ano, mes, dia_semana_name, tipo_crime, quantidade_ocorrencia`  
  (registros por delegacia).

- **dataset_encoded.csv:** contém as features one-hot codificadas:  
  - Ex.: `orgao_responsavel_Delegacia Pina`, `dia_semana_name_Monday`, `tipo_crime_Roubo`  
  - Além de `ano`, `mes` e target `quantidade_ocorrencia`.

- **modelo.ipynb:** notebook de treino dos modelos:  
  - RandomForestRegressor (`modelo_geral`) usando features codificadas.  
  - DecisionTreeRegressor (`modelo`) para comparação.  
  - Faz `train_test_split`, treino, avaliação (MAE/RMSE/R²) e plota reais × previstos.  
  - Exporta `dataset_encoded.csv` para uso no app.

**Conclusão:** O RandomForestRegressor é adequado para previsões agregadas por ano/mês/dia/delegacia/tipo de crime. A estratégia proposta é re-treinar o modelo ao carregar os dados ou usar um modelo salvo, gerando previsões agregadas para exibição no dashboard.

---

## Métricas de referência (teste reproduzido localmente)

RandomForestRegressor (`random_state=42`, `test_size=0.2`) sobre `dataset_encoded.csv`:

- MAE ≈ 0.33 (erro médio ~0.3 ocorrências por registro)  
- RMSE ≈ 0.40  
- R² ≈ 0.495 (modelo captura boa parte do padrão)

> Observação: O app exibirá as métricas reais do treino executado localmente.

---

## Requisitos

- Python 3.8+ (recomendado)  
- Virtualenv recomendado (não faça upload de `.venv` no repositório)

> ⚠️ **Boa prática:** Não subir `.venv` no repositório. Mantém o repo leve e evita conflitos.

---

## Instalação (Windows / PowerShell)

```powershell
cd path\to\Supervisionado

# criar virtualenv (opcional)
python -m venv .venv
.venv\Scripts\Activate.ps1

# instalar dependências
pip install -r requirements.txt
Execução
powershell
Copiar código
streamlit run app.py
Abra http://localhost:8501 no navegador.

Observações técnicas
Cálculo de RMSE compatível com várias versões do scikit-learn:

python
Copiar código
rmse = (mean_squared_error(...)) ** 0.5
Atualizar scikit-learn (opcional):

powershell
Copiar código
pip install -U scikit-learn
Problemas comuns
FileNotFoundError: verifique se dataset_encoded.csv e dataset_delegacias.txt estão na mesma pasta do app.py.

TypeError relacionado a squared: use a versão corrigida do train_model incluída ou atualize o scikit-learn.
