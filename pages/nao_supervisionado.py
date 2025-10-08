import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from nao_supervisionado.app import main

st.set_page_config(page_title="Análise Não Supervisionada", layout="wide")
main()