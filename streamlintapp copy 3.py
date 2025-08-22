import re
import io
import unicodedata
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt


st.set_page_config(page_title="Analise de Dados", layout="wide")

st.title("🔗 Resultado Forms Autopercepção - Meu código foi compartilhado com  alguns analistas para execução local -Entre em contato comigo:  Silvia Branco")
st.write("""
Este app:
1) **Permite a analise de dados de autopercepção** (CSV)
2) **Usa algoritmos de ML (K-Means)** para clusterização na base de respostas
3) **Gera gráficos interativos** para análise dos dados
  
""")
