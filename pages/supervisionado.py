import streamlit as st
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Supervisionado.app import main

st.set_page_config(page_title="Análise Supervisionada", layout="wide")
main()