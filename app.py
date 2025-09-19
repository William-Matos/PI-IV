import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import psycopg2 as pg
from warnings import filterwarnings
import folium
import numpy as np
import matplotlib.pyplot as plt
# from streamlit_folium import st_folium
from streamlit_folium import folium_static
import json

filterwarnings("ignore", category=UserWarning,
               message=".*pandas only supports SQLAlchemy connectable.*")

st.set_page_config(
    page_title='PI - IV: ENEM e Crescimento Econômico',
    layout='wide',
    page_icon=":line_chart:"
)

db_credenciais = st.secrets['postgresql']
host = db_credenciais['host']
port = db_credenciais['port']
user = db_credenciais['user']
password = db_credenciais['password']
dbname = db_credenciais['dbname']


@st.cache_data
def carregar_dados_agregados():
    with pg.connect(host=host, port=port, user=user, password=password, dbname=dbname) as conn:
        query_enem = """
            SELECT
                co_municipio_esc,
                AVG(nota_cn_ciencias_da_natureza) as media_cn,
                AVG(nota_ch_ciencias_humanas) as media_ch,
                AVG(nota_lc_linguagens_e_codigos) as media_lc,
                AVG(nota_mt_matematica) as media_mt,
                AVG(nota_redacao) as media_redacao
            FROM public.ed_enem_2024_resultados
            WHERE co_municipio_esc IS NOT NULL
            GROUP BY co_municipio_esc
        """
        enem_agregado = pd.read_sql_query(query_enem, conn)

        query_pib = """
            SELECT DISTINCT ON (codigo_municipio_dv)
                codigo_municipio_dv,
                vl_pib,
                vl_pib_per_capta
            FROM public.pib_municipios
            ORDER BY codigo_municipio_dv, ano_pib DESC
        """
        pib_municipios = pd.read_sql_query(query_pib, conn)

        query_censo = """
            SELECT
                "CO_MUNICIPIO",
                SUM("TOTAL") as pop_total,
                SUM(CASE WHEN "IDADE" BETWEEN 15 AND 19 THEN "TOTAL" ELSE 0 END) as pop_15_a_19
            FROM public."Censo_20222_Populacao_Idade_Sexo"
            GROUP BY "CO_MUNICIPIO"
        """
        censo_agregado = pd.read_sql_query(query_censo, conn)

        municipio = pd.read_sql_query(
            "SELECT nome_municipio, codigo_municipio_dv FROM public.municipio", conn)
        uf = pd.read_sql_query(
            "SELECT cd_uf, sigla_uf FROM public.unidade_federacao", conn)

    colunas_notas = ['media_cn', 'media_ch', 'media_lc', 'media_mt', 'media_redacao']
    enem_agregado['nota_media_geral'] = enem_agregado[colunas_notas].mean(axis=1)

    # Merge de ENEM com PIB
    df = pd.merge(
        enem_agregado,
        pib_municipios,
        left_on='co_municipio_esc',
        right_on='codigo_municipio_dv',
        how='left'
    )

    # Merge com dados do Censo
    df = pd.merge(
        df,
        censo_agregado,
        left_on='co_municipio_esc',
        right_on='CO_MUNICIPIO',
        how='left'
    )

    # Merge com nomes dos municípios
    df = pd.merge(
        df,
        municipio,
        left_on='co_municipio_esc',
        right_on='codigo_municipio_dv',
        how='left'
    )

    df['proporcao_jovem'] = df['pop_15_a_19'] / (df['pop_total'] + 1)

    df['cd_uf'] = df['co_municipio_esc'].astype(str).str[:2]
    uf.rename(columns={'sigla_uf': 'uf'}, inplace=True)

    df['cd_uf'] = df['cd_uf'].str.strip().replace('', np.nan)
    df.dropna(subset=['cd_uf'], inplace=True)
    df['cd_uf'] = df['cd_uf'].astype(int)
    uf['cd_uf'] = uf['cd_uf'].astype(int)

    df = pd.merge(
        df,
        uf[['cd_uf', 'uf']],
        on='cd_uf',
        how='left'
    )

    df = df.drop(columns=['codigo_municipio_dv_x',
                 'CO_MUNICIPIO', 'codigo_municipio_dv_y'])

    return df


df = carregar_dados_agregados()

valores = {'media_cn': df['media_cn'].mean(),
            'media_ch': df['media_cn'].mean(),
            'media_lc': df['media_lc'].mean(),
            'media_mt': df['media_mt'].mean(),
            'media_redacao': df['media_redacao'].mean(),
            'nota_media_geral': df['nota_media_geral'].mean()}

df.fillna(value=valores, inplace=True)
# st.info(df.shape)
# st.dataframe(df.head())
# st.info(
#     f"DataFrame final carregado com sucesso! Número de linhas: {len(df)}")

st.sidebar.title('Navegação')
pagina_selecionada = st.sidebar.radio(
    "Ir para",
    [
        '1. Apresentação do Projeto',
        '2. Análise Exploratória',
        '3. Análise Preditiva e Relatório',
        '4. Conclusão'
    ]
)
# =================================
# PÁGINA 1: APRESENTAÇÃO DO PROJETO
# =================================
if pagina_selecionada == '1. Apresentação do Projeto':
    st.title("Relação entre Desempennho no ENEM e Crescimento Econômico Municipal")
    st.markdown("---")

    st.header('1. Problema de Pesquisa e Contextualização')
    st.write("""
        Este projeto busca responder à seguinte pergunta: **Há relação entre o desempenho 
        dos estudantes do ensino médio e o crescimento econômico nos municípios brasileiros?**
             
        A educação é frequentemente citada como um pilar para o desenvolvimento socioeconômico de um país.
        Analisar dados a nível municipal permite identificar padrões que podem passar despercebidos em 
        análises mais amplas, a nível estadual ou nacional.
        """)

    st.subheader('Variáveis de Análise')
    st.markdown("""
        * **Variável Target:**
                *`PIB per capita`: Produto Interno Bruto do município divido por sua população total.
        * **Variável Independente Principal:**
            * `Nota Média ENEM`: Média das notas dos participantes do ENEM para cada município. Serve como proxy para a qualidade da educação de nível médio.
        * **Variáveis de Controle (Demográficas):**
            * `Proporção da População Jovem`: Percentual da população entre 15 e 19 anos.
            * -- Variáveis a decidir --.
        """)

    st.markdown("---")
    st.header('2. Metodologia e Modelos Analíticos')
    st.write("""
        Para investigar a relação proposta, utilizamos uma abordagem complementar com dois modelos analíticos distintos:
        """)

    st.subheader('A. Regressão Linear Múltipla')
    st.write(r"""
        **Justificativa:** Escolhido por sua alta **interpretabilidade**. Este modelo nos permite quantificar 
        diretamente a relação entre a nota média do ENEM e o PIB per capita, mostrando o quanto o PIB tende a variar 
        para cada ponto a mais na nota do ENEM, enquanto se controla por outros fatores demográficos.
        
        A fórmula básica investigada é: 
        $$
        PIB_{per\_capita} = \\beta_0 + \\beta_1 \\cdot NotaMédia\_{ENEM} + \\beta_2 \\cdot VarControle + \\epsilon
        $$
    """)

    st.subheader("B. Random Forest Regressor")
    st.write("""
        **Justificativa:** Selecionado por seu alto **poder preditivo** e capacidade de capturar relações complexas 
        e não-lineares. Enquanto a regressão linear nos diz *qual* a relação, o Random Forest nos diz *quão bem* podemos prever o desenvolvimento econômico de um município com base em suas características educacionais 
        e demográficas. Além disso, ele nos fornece a **importância das variáveis** (feature importance), 
        indicando quais fatores são mais relevantes para a predição.
    """)

# =================================
# PÁGINA 2: ANÁLISE EXPLORATÓRIA
# =================================
elif pagina_selecionada == "2. Análise Exploratória":
    st.sidebar.header('Filtros para Análise')

    lista_ufs = df['uf'].unique()

    selecionar_todas = st.sidebar.checkbox('Selecionar Todas as UFs', value=True)
    
    ufs_padroes = lista_ufs if selecionar_todas else []

    ufs_selecionadas = st.sidebar.multiselect(
        'Selecione a UF',
        options=lista_ufs,
        default=ufs_padroes
    )

    df_filtrado = df[df['uf'].isin(ufs_selecionadas)]

    tab1, tab2, tab3 = st.tabs(["🗺️ Mapa Interativo", "📊 Relação entre Variáveis", "📈 Distribuições"])

    with tab1:
        st.subheader('Distribuição Geográfica das Variáveis')

        variavel_mapa = st.selectbox(
            "Selecione a variável para visualizar no mapa:",
            ['vl_pib_per_capta', 'nota_media_geral'],
            index=None,
            format_func=lambda x: 'PIB per Capta' if x == 'vl_pib_per_capta' else 'Nota Média ENEM'
        )
        if variavel_mapa:
            st.info("Passe o mouse sobre os municípios para ver os valores.")

            path_geojson = 'geojs-100-mun.json'
            
            @st.cache_data
            def load_geojson(path):
                with open(path_geojson, 'r', encoding='utf-8') as f:
                    return json.load(f)
                
            geojson_data = load_geojson(path_geojson)

            df_filtrado['co_municipio_esc_str'] = df_filtrado['co_municipio_esc'].astype(str)
            data_dict = df_filtrado.set_index('co_municipio_esc_str').to_dict('index')

            for feature in geojson_data['features']:
                mun_id = feature['properties']['id']
                if mun_id in data_dict:
                    dados = data_dict[mun_id]
                    feature['properties']['nome_municipio'] = dados.get('nome_municipio', 'N/A')
                    feature['properties']['uf'] = dados.get('uf', 'N/A')
                    
                    pib_formatado = f"R$ {dados.get('vl_pib_per_capta', 0):,.2f}"
                    nota_formatada = f"{dados.get('nota_media_geral', 0):.2f}"
                    
                    feature['properties']['pib_formatado'] = pib_formatado
                    feature['properties']['nota_formatada'] = nota_formatada
                else: 
                    feature['properties']['nome_municipio'] = 'Dado não disponível'
                    feature['properties']['uf'] = ''
                    feature['properties']['pib_formatado'] = 'N/A'
                    feature['properties']['nota_formatada'] = 'N/A'
            
            bins = list(df_filtrado[variavel_mapa].quantile(np.linspace(0, 1, 8)))

            mapa = folium.Map(location=[-14.2350, -51.9253], zoom_start=4)

            choropleth = folium.Choropleth(
                geo_data=geojson_data,
                data=df_filtrado,
                columns=['co_municipio_esc_str', variavel_mapa],
                key_on='feature.properties.id',
                fill_color='YlOrRd',
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name=f'Valor de {variavel_mapa}',
                bins=bins,
                highlight=True
            ).add_to(mapa)


            tooltip = folium.GeoJsonTooltip(
                fields=['nome_municipio', 'uf', 'pib_formatado', 'nota_formatada'],
                aliases=['Município:', 'UF:', 'PIB per Capita:', 'Nota Média ENEM:'],
                sticky=True,
                style=("background-color: white; color: black; font-family: arial; font-size: 12px; padding: 10px;")
            )

            folium.GeoJson(
                geojson_data,
                style_function=lambda x: {'fillOpacity': 0, 'color': 'transparent', 'weight': 0}, 
                tooltip=tooltip
            ).add_to(mapa)
        
            folium_static(mapa, width=None, height=500)

    with tab2:
        st.subheader('Correlação entre Nota do ENEM e PIB per Capita')
        fig_scatter = px.scatter(
            df_filtrado,
            x='nota_media_geral',
            y='vl_pib_per_capta',
            hover_data=['nome_municipio', 'uf'],
            labels={
                'nota_media_enem': 'Nota Média no ENEM',
                'pib_per_capita': 'PIB per Capita (R$)'
            },
            title='PIB per Capita vs. Nota Média no ENEM'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader('Correlação (com Eixo Y em Escala Logarítmica)')
        st.write("Aplicar uma escala logarítmica no eixo do PIB ajuda a reduzir o efeito dos outliers e pode revelar tendências que estavam escondidas.")

        fig_scatter_log = px.scatter(
            df_filtrado,
            x='nota_media_geral',
            y='vl_pib_per_capta',
            log_y=True,  # <-- A MÁGICA ACONTECE AQUI
            hover_data=['nome_municipio', 'uf'],
            labels={
                'nota_media_geral': 'Nota Média no ENEM',
                'vl_pib_per_capta': 'PIB per Capita (R$ - Escala Log)'
            },
            title='PIB per Capita (Log) vs. Nota Média no ENEM'
        )
        st.plotly_chart(fig_scatter_log, use_container_width=True)

    with tab3:
        st.subheader('Dsitribuiçãoi das Variáveis Chave')
        col1, col2 = st.columns(2)
        with col1:
            fig_hist_enem = px.histogram(
                df_filtrado,
                x='nota_media_geral',
                nbins=50,
                title='Distribuição das Notas Médias do ENEM'
            )
            st.plotly_chart(fig_hist_enem)
        with col2:
            fig_hist_pib = px.histogram(
                df_filtrado, x='vl_pib_per_capta', nbins=50, title='Distribuição do PIB per Capta')
            st.plotly_chart(fig_hist_pib)

# =======================================
# PÁGINA 3: ANÁLISE PREDITIVA E RELATÓRIO
# =======================================
elif pagina_selecionada == "3. Análise Preditiva e Relatório":
    st.header("Modelagem Preditiva: Prevendo o PIB per capita")
    st.write("""
        Nesta seção, treinei dois modelos de regressão para prever o PIB per capita
        de um município com base em suas características educacionais e demográficas.
        
        **Importante:** A variável PIB per capita possui uma distribuição muito assimétrica 
        (poucos municípios são muito ricos). Para melhorar o desempenho dos modelos, 
        aplicamos uma **transformação logarítmica** (log(1+x)) sobre ela. 
        Isso ajuda a normalizar a distribuição e a estabilizar a variância.
    """)

    df.rename(columns={
        'media_cn': 'Media_Ciencias_Natureza',
        'media_ch': 'Media_Ciencias_Humanas',
        'media_lc': 'Media_Linguagens_Codigos',
        'media_mt': 'Media_Matematica',
        'media_redacao': 'Media_Redacao'
    }, inplace=True)
    
    features = [
        'nota_media_geral',
        'proporcao_jovem',
        'pop_total',
        'Media_Ciencias_Natureza',
        'Media_Ciencias_Humanas',
        'Media_Linguagens_Codigos',
        'Media_Matematica',
        'Media_Redacao'
    ]
    df_modelo = df.dropna(subset=features + ['vl_pib_per_capta'])
    X = df_modelo[features]

    y = np.log1p(df_modelo['vl_pib_per_capta'])
    # y = df_modelo['vl_pib_per_capta']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tab1_p3, tab2_p3, tab3_p3 = st.tabs(['Regressão Linear', 'Random Forest', 'Árvore de Decisão'])

    with tab1_p3:
        st.header('Treinamento do Modelo de Regressão Linear')
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)

        y_pred_lr = lr_model.predict(X_test)
        r2_lr = r2_score(y_test, y_pred_lr)
        rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

        # st.write(f"Coeficients: {lr_model.coef_}")
        st.metric(label="R² (Regressão Linear)", value=f"{r2_lr:.3f}")
        st.metric(label="RMSE (Regressão Linear)", value=f"{rmse_lr:.3f}")

        st.subheader("Interpretando os Coeficientes do Modelo")
        
        coefs = pd.DataFrame(lr_model.coef_, index=features, columns=['Coeficiente'])
        coefs['Impacto'] = coefs['Coeficiente'].apply(lambda x: 'Positivo' if x > 0 else 'Negativo')
        
        st.dataframe(coefs.sort_values(by='Coeficiente', ascending=False))
        
        st.info("""
            **Como ler esta tabela:** Um coeficiente positivo significa que, mantendo as outras variáveis constantes, um aumento nesta variável tende a aumentar o log do PIB per capita. O oposto é verdadeiro para coeficientes negativos.
        """)
        st.write("---") # Adiciona uma linha divisória

        # --- Gráfico de Análise de Resíduos (Previsto vs. Real) ---
        st.subheader('Análise Gráfica do Modelo de Regressão Linear')
        st.write("""
            O gráfico abaixo compara os **valores reais** do log do PIB per capita (no eixo X)
            com os **valores previstos** pelo modelo (no eixo Y).
            
            Se o modelo fosse perfeito, todos os pontos estariam sobre a linha vermelha tracejada.
            Quanto mais próximos os pontos estiverem dessa linha, melhores são as previsões.
        """)

        plot_df = pd.DataFrame({
            'Valores Reais (log)': y_test,
            'Valores Previstos (log)': y_pred_lr
        })

        fig_scatter = px.scatter(
            plot_df,
            x='Valores Reais (log)',
            y='Valores Previstos (log)',
            title='Valores Previstos vs. Valores Reais (Regressão Linear)',
            labels={'Valores Reais (log)': 'Valores Reais (log PIB per capita)', 'Valores Previstos (log)': 'Valores Previstos pelo Modelo (log)'},
            hover_data={'Valores Reais (log)': ':.2f', 'Valores Previstos (log)': ':.2f'} # Formata o hover
        )

        min_val = min(y_test.min(), y_pred_lr.min())
        max_val = max(y_test.max(), y_pred_lr.max())
        fig_scatter.add_shape(
            type='line',
            x0=min_val, y0=min_val,
            x1=max_val, y1=max_val,
            line=dict(color='Red', dash='dash')
        )

        st.plotly_chart(fig_scatter, use_container_width=True)
        

    with tab2_p3:
        st.header("Treinamento do Modelo Random Forest")
        
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)

        y_pred_rf = rf_model.predict(X_test)
        r2_rf = r2_score(y_test, y_pred_rf)
        rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

        st.metric(label="R² (Random Forest)", value=f"{r2_rf:.3f}")
        st.metric(label="RMSE (Random Forest)", value=f"{rmse_rf:.3f}")
        st.subheader("Importância das Variáveis (Feature Importance)")
        st.write("O gráfico abaixo mostra quais variáveis o modelo Random Forest considerou mais importantes para fazer as previsões.")
        feature_importances = pd.DataFrame(rf_model.feature_importances_, index=features, columns=['importance'])
        st.bar_chart(feature_importances.sort_values(by='importance', ascending=False))

    with tab3_p3:
        st.header("Análise do Modelo de Árvore de Decisão")
        st.write("""
            Enquanto o Random Forest usa centenas de árvores, analisar uma única árvore
            nos ajuda a entender como o modelo toma decisões. 
            Abaixo, avaliamos e visualizamos uma árvore simples (limitada a 3 níveis de profundidade) para ilustrar o processo.
        """)
        
        # Treinamos o modelo de árvore
        tree_model = tree.DecisionTreeRegressor(max_depth=3, random_state=42)
        tree_model.fit(X_train, y_train)

        # PASSO 1: Fazer previsões no conjunto de teste
        y_pred_tree = tree_model.predict(X_test)

        # PASSO 2: Calcular as métricas de regressão
        r2_tree = r2_score(y_test, y_pred_tree)
        rmse_tree = np.sqrt(mean_squared_error(y_test, y_pred_tree))

        st.subheader("Métricas de Desempenho")
        
        # PASSO 3: Exibir as métricas
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="R² (Árvore de Decisão)", value=f"{r2_tree:.3f}")
        with col2:
            st.metric(label="RMSE (Árvore de Decisão)", value=f"{rmse_tree:.3f}")
        
        st.info("""
            **Observação:** É esperado que o R² de uma única árvore seja **menor** que o do Random Forest.
            O Random Forest combina a "sabedoria" de muitas árvores para criar uma previsão mais robusta e precisa.
        """)
        
        st.markdown("---")

        st.subheader("Visualização da Árvore")
        # O código para plotar a árvore continua o mesmo
        fig, ax = plt.subplots(figsize=(20, 10))
        tree.plot_tree(
            tree_model,
            feature_names=features,
            filled=True,
            rounded=True,
            fontsize=10,
            ax=ax
        )
        st.pyplot(fig)

# ===================
# PÁGINA 4: CONCLUSÃO
# ===================
elif pagina_selecionada == "4. Conclusão":
    st.title('Relatório Analítico e Conclusão')
    st.markdown('---')

    # Passo 1: Preparar os dados e treinar o modelo final
    # Usamos os mesmos dados preparados da página 3
    df.rename(columns={
        'media_cn': 'Media_Ciencias_Natureza',
        'media_ch': 'Media_Ciencias_Humanas',
        'media_lc': 'Media_Linguagens_Codigos',
        'media_mt': 'Media_Matematica',
        'media_redacao': 'Media_Redacao'
    }, inplace=True)

    features = [
        'nota_media_geral',
        'proporcao_jovem',
        'pop_total',
        'Media_Ciencias_Natureza',
        'Media_Ciencias_Humanas',
        'Media_Linguagens_Codigos',
        'Media_Matematica',
        'Media_Redacao'
    ]
    
    df_modelo = df.dropna(subset=features + ['vl_pib_per_capta']).copy()
    
    X_full = df_modelo[features]
    y_full_log = np.log1p(df_modelo['vl_pib_per_capta'])

    rf_final_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_final_model.fit(X_full, y_full_log)

    predicoes_log = rf_final_model.predict(X_full)
    
    df_modelo['pib_predito'] = np.expm1(predicoes_log)
    df_modelo['diferenca_predicao'] = df_modelo['pib_predito'] - df_modelo['vl_pib_per_capta']

    st.header('Análise Interativa por Município')
    st.write('Selecione um município para ver seus dados e a predição do modelo de Random Forest.')

    df_modelo['display_name'] = df_modelo['nome_municipio'] + ' - ' + df_modelo['uf']
    
    municipio_selecionado = st.selectbox(
        "Selecione o Município",
        options=df_modelo['display_name'].sort_values()
    )

    if municipio_selecionado:
        dados_municipio = df_modelo[df_modelo['display_name'] == municipio_selecionado].iloc[0]

        st.subheader(f"Resultados para {municipio_selecionado}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="PIB per Capita Real",
                value=f"R$ {dados_municipio['vl_pib_per_capta']:,.2f}"
            )
        with col2:
            st.metric(
                label="PIB per Capita PREVISTO",
                value=f"R$ {dados_municipio['pib_predito']:,.2f}"
            )
        with col3:
            st.metric(
                label="Diferença (Previsto - Real)",
                value=f"R$ {dados_municipio['diferenca_predicao']:,.2f}",
                delta_color="inverse" # Fica vermelho para valores negativos
            )
        
        st.markdown("---")
        st.subheader("Dados do Município Utilizados no Modelo")
        
        st.table(dados_municipio[features].rename("Valor").to_frame())