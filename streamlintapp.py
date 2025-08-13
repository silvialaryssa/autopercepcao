import re
import io
import unicodedata
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt


st.set_page_config(page_title="Analise de Dados", layout="wide")

st.title("🔗 Resultado Forms Autopercepção")
st.write("""
Este app:
1) **Permite a analise de dados de autopercepção** (CSV)
2) **Usa algoritmos de ML (K-Means)** para clusterização na base de respostas
3) **Gera gráficos interativos** para análise dos dados
  
""")

################################################################
# arquivo merge_matriculas.py

# ========= Upload =========
st.subheader("1) Merge Matrículas")
#c_up1 = st.columns(1)
#with c_up1:
arq_merge = st.file_uploader("📤 merge_matriculas", type=["csv", "CSV"], key="auto")

# ========= Análise e Resumo por Chave =========
if arq_merge is None:
    st.error("Por favor, faça o upload do arquivo 'merge_matriculas.csv'.")
    st.stop()
st.subheader("🔍 Análise e Resumo por Chave")
#carrega o CSV do merge
#df_merge = pd.read_csv("merge_matriculas.csv", dtype=str)
df_merge = pd.read_csv(arq_merge, dtype=str)
if 'df_merge' not in locals():
    st.error("Por favor, faça o merge primeiro execuntando o arquivo app.py.")
    st.stop()
df = df_merge.copy()


# Garantir que a coluna 'matricula' existe e limpar espaços
df.columns = df.columns.str.strip().str.lower()

# Agrupar por matricula e calcular a média (pode trocar para soma, mediana, etc.)
#df_grouped = df.groupby("matricula").mean(numeric_only=True).reset_index()

st.dataframe(df.sort_values("matricula"), use_container_width=True)

# 1) pega só as colunas de notas (nomes que começam com '[')
cols = [c for c in df.columns if str(c).lstrip().startswith('[')]

#exibir as colunas cols
#st.write("**Colunas de notas disponíveis:**")
#st.write(cols)

# 2) garante que são numéricas (simples)
for c in cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

ranking_geral = (
    df[cols].mean(skipna=True)
      .sort_values(ascending=False)
      .rename('media')
      .reset_index()
      .rename(columns={'index':'pergunta'})
)

# exibir o numero de linhas do ranking geral
st.write(f"**Ranking geral (média por pergunta):** {len(ranking_geral)} perguntas")
st.subheader("🏆 Ranking geral (todas as perguntas)")
st.dataframe(ranking_geral, use_container_width=True)




# --- Ranking geral por grupo (linhagerencia / squadtime / papel / funcao) ---
# colunas de nota (começam com '[')
# exibir a matricula
# converte as colunas de notas para numéricas (se não forem)
cols = [c for c in df.columns if str(c).lstrip().startswith('[')]
for c in cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')


# gera um ranking separado para cada chave, exibindo a matrícula
for chave in ['linhagerencia', 'squadtime', 'papel', 'funcao']:
    if chave in df.columns and 'matricula' in df.columns:
        ranking = (
            df.groupby([chave, 'matricula'], dropna=False)[cols].mean().reset_index()
              .melt(id_vars=[chave, 'matricula'], var_name='pergunta', value_name='media')
              .sort_values([chave, 'matricula', 'media'], ascending=[True, True, False])
        )
        st.subheader(f"🏆 Ranking por {chave} (com matrícula)")
        st.dataframe(ranking, use_container_width=True)

# --- Gráfico: Ranking geral (média por pergunta) agrupado por linhagerencia, squadtime, papel, funcao ---
st.subheader("📊 Gráfico: Ranking geral (média por pergunta)"
              )
# Verifica se as colunas de notas existem
if not cols:    
    st.warning("Nenhuma coluna de notas encontrada. Verifique os nomes das colunas.")
else:
    # Calcula a média por pergunta
    ranking_geral = (
        df[cols].mean(skipna=True)
          .sort_values(ascending=False)
          .rename('media')
          .reset_index()
          .rename(columns={'index':'pergunta'})
    )

# exibir um grafico de barras horizontal de medias por pergunta agrupador por linhagerencia apenas
# linhasgerencia

# --- Média geral por linhagerencia (Altair interativo) ---
if 'linhagerencia' in df.columns:
    base = df.copy()
    base['linhagerencia'] = base['linhagerencia'].fillna('Sem linhagerencia').astype(str).str.strip()

    media_geral_lg = (
        base.groupby('linhagerencia')[cols].mean()   # média por pergunta no grupo
            .mean(axis=1)                            # média geral (todas as perguntas)
            .sort_values(ascending=False)
            .reset_index(name='media_geral')
    )

    # Top N para não poluir o gráfico
    max_n = int(min(50, len(media_geral_lg)))
    n = st.slider("Top N (linhagerencia)", 5, max_n, min(10, max_n))
    plot_df = media_geral_lg.head(n)

    # seleção para destacar no hover/click
    sel = alt.selection_point(fields=['linhagerencia'], on='mouseover', empty='none')
    cor =  "#55A84C"

    bars = (
        alt.Chart(plot_df)
           .mark_bar(color=cor)
           .encode(
               y=alt.Y('linhagerencia:N', sort='-x', title='Linhagerência'),
               x=alt.X('media_geral:Q', title='Média geral'),
               tooltip=['linhagerencia', alt.Tooltip('media_geral:Q', format='.2f')],
               opacity=alt.condition(sel, alt.value(1), alt.value(0.5))
           )
           .add_params(sel)
           .properties(height=28 * len(plot_df) + 30, title='Média geral por linhagerencia')
           .interactive()
    )

    # rótulos de valor na ponta das barras
    labels = (
        alt.Chart(plot_df)
           .mark_text(align='left', dx=3)
           .encode(
               y=alt.Y('linhagerencia:N', sort='-x'),
               x=alt.X('media_geral:Q'),
               text=alt.Text('media_geral:Q', format='.2f')
           )
    )

    st.altair_chart(bars + labels, use_container_width=True)
else:
    st.warning("Coluna 'linhagerencia' não encontrada no dataframe.")



for chave, titulo in [('squadtime', 'SquadTime'), ('papel', 'Papel'), ('funcao', 'Função')]:
    if chave in df.columns:
        base = df.copy()
        base[chave] = base[chave].fillna(f'Sem {titulo}').astype(str).str.strip()

        # média das perguntas no grupo -> média geral do grupo
        media = (
            base.groupby(chave)[cols].mean()   # média por pergunta dentro do grupo
                .mean(axis=1)                  # média geral (todas as perguntas)
                .sort_values(ascending=False)
                .reset_index(name='media_geral')
        )

        if media.empty:
            st.info(f"Sem dados para {titulo}.")
            continue

        # Top N com proteção para poucos grupos
        max_n = max(1, min(50, len(media)))
        n = st.slider(f"Top N ({titulo})", 1, max_n, min(10, max_n), key=f"slider_{chave}")
        plot_df = media.head(n)

        sel = alt.selection_point(fields=[chave], on='mouseover', empty='none')
        #cor = st.color_picker(f"Cor das barras ({titulo})", "#55A84C", key=f"cor_{chave}")
        cor = "#55A84C"  # cor fixa para as barras
        bars = (
            alt.Chart(plot_df)
               .mark_bar(color=cor)
               .encode(
                   y=alt.Y(f'{chave}:N', sort='-x', title=titulo),
                   x=alt.X('media_geral:Q', title='Média geral'),
                   tooltip=[chave, alt.Tooltip('media_geral:Q', format='.2f')],
                   opacity=alt.condition(sel, alt.value(1), alt.value(0.5))
               )
               .add_params(sel)
               .properties(height=28 * len(plot_df) + 30, title=f'Média geral por {titulo}')
               .interactive()
        )

        labels = (
            alt.Chart(plot_df)
               .mark_text(align='left', dx=3)
               .encode(
                   y=alt.Y(f'{chave}:N', sort='-x'),
                   x=alt.X('media_geral:Q'),
                   text=alt.Text('media_geral:Q', format='.2f')
                   
               )
        )

        st.subheader(f"📊 Média geral por {titulo}")
        st.altair_chart(bars + labels, use_container_width=True)


st.subheader("🏆 Ranking geral (média por linhagerencia, squadtime, papel, funcao)")
ranking_geral = (
    df[cols].mean(skipna=True)
      .sort_values(ascending=False)
      .rename('media')
      .reset_index()
      .rename(columns={'index':'pergunta'})
)

        # --- Gráfico: Ranking geral (Top N) ---


max_n = int(min(30, len(ranking_geral)))
n = st.slider("Top N para o gráfico (ranking geral)", 5, max_n, min(10, max_n))

plot_data = ranking_geral.head(n).iloc[::-1]  # inverte p/ barh de cima p/ baixo
fig, ax = plt.subplots(figsize=(10, 0.5*n + 1))
ax.barh(plot_data['pergunta'], plot_data['media'])
ax.set_xlabel("Média")
ax.set_ylabel("Pergunta")
ax.set_title("Ranking geral - Top N")
plt.tight_layout()
st.pyplot(fig)


# --- Gráfico: Ranking por grupo (perguntas) ---
keys_disp = [k for k in ['linhagerencia', 'squadtime', 'papel', 'funcao'] if k in df.columns]
if keys_disp:
    c1, c2 = st.columns(2)
    with c1:
        chave = st.selectbox("Agrupar por", keys_disp, index=0)
    with c2:
        valores = sorted(df[chave].dropna().unique().tolist())
        if len(valores) == 0:
            st.info(f"Sem valores em {chave}.")
        else:
            valor = st.selectbox(f"Selecionar {chave}", valores, index=0)

            medias = (
                df.loc[df[chave] == valor, cols]
                  .mean(skipna=True)
                  .sort_values(ascending=False)
                  .rename('media')
                  .reset_index()
                  .rename(columns={'index':'pergunta'})
            )

            st.markdown(f"### 📊 {chave}: **{valor}** — ranking de perguntas")
            n2 = min(15, len(medias))
            fig2, ax2 = plt.subplots(figsize=(10, 0.5*n2 + 1))
            plot2 = medias.head(n2).iloc[::-1]
            ax2.barh(plot2['pergunta'], plot2['media'])
            ax2.set_xlabel("Média")
            ax2.set_ylabel("Pergunta")
            ax2.set_title(f"Top {n2} — {chave}: {valor}")
            plt.tight_layout()
            st.pyplot(fig2)

# --- Gráfico: Top matrículas no grupo selecionado (média global das notas) ---
if 'matricula' in df.columns and len(valores) > 0:
    df_grp = df.loc[df[chave] == valor].copy()
    if not df_grp.empty:
        score_matricula = (
            df_grp.groupby('matricula')[cols].mean().mean(axis=1).sort_values(ascending=False)
        )

        top_m = min(15, len(score_matricula))
        st.markdown(f"### 🧑‍💼 Top matrículas — {chave}: **{valor}**")
        fig3, ax3 = plt.subplots(figsize=(8, 0.4*top_m + 1))
        plot3 = score_matricula.head(top_m).iloc[::-1]
        ax3.barh(plot3.index.astype(str), plot3.values)
        ax3.set_xlabel("Média geral (todas as perguntas)")
        ax3.set_ylabel("Matrícula")
        ax3.set_title(f"Top {top_m} matrículas — {chave}: {valor}")
        plt.tight_layout()
        st.pyplot(fig3)


###############################################################################################

# =========================
# ML (A) CLUSTERIZAÇÃO
# =========================
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import altair as alt

st.subheader("🤖 Clusterização (K-Means)  -  FAIXAS DE NOTAS")

if len(cols) == 0:
    st.warning("Não encontrei colunas de notas (as que começam com '[').")
else:
    # Features: notas
    X = df[cols].copy()
    # Imputação simples
    X = X.fillna(X.mean(numeric_only=True))
    # Padroniza
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k = st.slider("Número de clusters (k)", 2, 5, 3)
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X_scaled)

    # Silhouette (qualidade do agrupamento)
    sil = silhouette_score(X_scaled, labels)
    st.write(f"**Silhouette:** {sil:.3f} (quanto mais perto de 1, melhor separação)")

    # Anexa cluster no df para exploração
    df["_cluster"] = labels

    # Perfil dos clusters (média de cada pergunta)
    perfil = df.groupby("_cluster")[cols].mean().round(2)
    st.markdown("**Média das perguntas por cluster**")
    st.dataframe(perfil, use_container_width=True)

    # PCA p/ plot 2D
    pca = PCA(n_components=2, random_state=42)
    comp = pca.fit_transform(X_scaled)
    plot_df = pd.DataFrame({
        "pc1": comp[:, 0], "pc2": comp[:, 1], "_cluster": labels.astype(str)
    })
    # Altair scatter interativo
    sel = alt.selection_point(fields=["_cluster"], on="mouseover", empty="all")
    chart = (
        alt.Chart(plot_df)
           .mark_circle(size=60)
           .encode(
               x=alt.X("pc1:Q", title="PC1"),
               y=alt.Y("pc2:Q", title="PC2"),
               color=alt.Color("_cluster:N", title="Cluster"),
               tooltip=["_cluster", "pc1", "pc2"],
               opacity=alt.condition(sel, alt.value(1), alt.value(0.4))
           )
           .add_params(sel)
           .properties(height=400, title="Clusters (PCA 2D)")
           .interactive()
    )
    #st.altair_chart(chart, use_container_width=True)

# média geral por pessoa (média de todas as perguntas)
df['_score_geral'] = df[cols].mean(axis=1)

resumo = (
    df.groupby('_cluster')['_score_geral']
      .agg(qtd='count', media='mean')
      .reset_index()
      .sort_values('media', ascending=False)
)
st.subheader("📋 Resumo por cluster")
st.dataframe(resumo, use_container_width=True)


import altair as alt

perfil = (
    df.groupby('_cluster')[cols].mean().reset_index()
      .melt(id_vars=['_cluster'], var_name='pergunta', value_name='media')
)

st.subheader("🔥 Perfil dos clusters por pergunta (média)")
heat = (
    alt.Chart(perfil)
       .mark_rect()
       .encode(
           x=alt.X('pergunta:N', sort=None, title='Pergunta'),
           y=alt.Y('_cluster:N', title='Cluster'),
           color=alt.Color('media:Q', scale=alt.Scale(scheme='blues')),
           tooltip=['_cluster', 'pergunta', alt.Tooltip('media:Q', format='.2f')]
       )
       .properties(height=200)
       .interactive()
)
st.altair_chart(heat, use_container_width=True)

#TOP_N = 5


#top_por_cluster = (
#    df.groupby('_cluster')[cols].mean()          # média por pergunta em cada cluster
#      .stack()                                   # vira Series com MultiIndex
#      .rename_axis(['_cluster', 'pergunta'])     # nomeia os níveis do índice
#      .reset_index(name='media')                 # <- name (sem S)
#      .sort_values(['_cluster', 'media'], ascending=[True, False])
#      .groupby('_cluster', as_index=False)
#      .head(TOP_N)
#)

#st.subheader(f"🏆 Top {TOP_N} perguntas por cluster")
#st.dataframe(top_por_cluster, use_container_width=True)
