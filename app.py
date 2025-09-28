import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.metrics import mean_squared_error, r2_score
import psycopg2 as pg
from warnings import filterwarnings
import folium
import numpy as np
import matplotlib.pyplot as plt
from streamlit_folium import folium_static
import json

filterwarnings("ignore", category=UserWarning,
               message=".*pandas only supports SQLAlchemy connectable.*")

st.set_page_config(
    page_title='PI - IV: ENEM e Crescimento Econômico',
    layout='wide',
    page_icon=":line_chart:"
)

# =================================
# CARREGAMENTO E CACHE DOS DADOS
# =================================
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
                codigo_municipio_dv, vl_pib, vl_pib_per_capta
            FROM public.pib_municipios
            ORDER BY codigo_municipio_dv, ano_pib DESC
        """
        pib_municipios = pd.read_sql_query(query_pib, conn)

        query_censo = """
            SELECT
                "CO_MUNICIPIO", SUM("TOTAL") as pop_total,
                SUM(CASE WHEN "IDADE" BETWEEN 15 AND 19 THEN "TOTAL" ELSE 0 END) as pop_15_a_19
            FROM public."Censo_20222_Populacao_Idade_Sexo"
            GROUP BY "CO_MUNICIPIO"
        """
        censo_agregado = pd.read_sql_query(query_censo, conn)

        municipio = pd.read_sql_query(
            "SELECT nome_municipio, codigo_municipio_dv FROM public.municipio", conn)
        
        uf = pd.read_sql_query(
            "SELECT cd_uf, sigla_uf FROM public.unidade_federacao", conn)

        query_features_escola = """
            SELECT
                co_municipio_esc,
                COUNT(*) AS total_alunos,
                SUM(CASE WHEN tp_dependencia_adm_esc = 'Privada' THEN 1 ELSE 0 END) AS alunos_privada,
                SUM(CASE WHEN tp_localizacao_esc = 'Urbana' THEN 1 ELSE 0 END) AS alunos_urbana,
                SUM(CASE WHEN tp_lingua = 'Inglês' THEN 1 ELSE 0 END) AS alunos_ingles,
                SUM(CASE WHEN tp_status_redacao = 'Em Branco' THEN 1 ELSE 0 END) AS redacoes_em_branco
            FROM public.ed_enem_2024_resultados
            WHERE co_municipio_esc IS NOT NULL
            GROUP BY co_municipio_esc
        """
        features_escola = pd.read_sql_query(query_features_escola, conn)

    # PREPARAÇÃO E ENGENHARIA DE FEATURES
    colunas_notas = ['media_cn', 'media_ch', 'media_lc', 'media_mt', 'media_redacao']
    enem_agregado['nota_media_geral'] = enem_agregado[colunas_notas].mean(axis=1)
    
    features_escola['perc_privada'] = (features_escola['alunos_privada'] / features_escola['total_alunos']) * 100
    features_escola['perc_urbana'] = (features_escola['alunos_urbana'] / features_escola['total_alunos']) * 100
    features_escola['perc_ingles'] = (features_escola['alunos_ingles'] / features_escola['total_alunos']) * 100
    features_escola['perc_redacoes_branco'] = (features_escola['redacoes_em_branco'] / features_escola['total_alunos']) * 100

    df = enem_agregado.copy()
    df = pd.merge(df, features_escola[['co_municipio_esc', 'perc_privada', 'perc_urbana', 'perc_ingles', 'perc_redacoes_branco']], on='co_municipio_esc', how='left')
    df = pd.merge(df, pib_municipios, left_on='co_municipio_esc', right_on='codigo_municipio_dv', how='left')
    df = pd.merge(df, censo_agregado, left_on='co_municipio_esc', right_on='CO_MUNICIPIO', how='left')
    df = pd.merge(df, municipio, left_on='co_municipio_esc', right_on='codigo_municipio_dv', how='left')
    
    df[['perc_privada', 'perc_urbana']] = df[['perc_privada', 'perc_urbana']].fillna(0)
    
    df['proporcao_jovem'] = df['pop_15_a_19'] / (df['pop_total'] + 1)
    df['cd_uf'] = df['co_municipio_esc'].astype(str).str[:2]
    uf.rename(columns={'sigla_uf': 'uf'}, inplace=True)

    df['cd_uf'] = df['cd_uf'].str.strip().replace('', np.nan)
    df.dropna(subset=['cd_uf'], inplace=True)
    df['cd_uf'] = df['cd_uf'].astype(int)
    uf['cd_uf'] = uf['cd_uf'].astype(int)

    df = pd.merge(df, uf[['cd_uf', 'uf']], on='cd_uf', how='left')

    df = df.drop(columns=['codigo_municipio_dv_x', 'CO_MUNICIPIO', 'codigo_municipio_dv_y', 'codigo_municipio_dv'], errors='ignore')

    return df

# =================================
# FUNÇÃO DE PRÉ-PROCESSAMENTO PARA MODELO
# =================================
def preprocessar_para_modelo(df_input):
    df_processed = df_input.copy()
    
    # Engenharia de Features
    df_processed['log_pop_total'] = np.log1p(df_processed['pop_total'])
    df_processed['nota_x_proporcao_jovem'] = df_processed['nota_media_geral'] * df_processed['proporcao_jovem']
    
    # One-Hot Encoding para a UF
    df_processed = pd.get_dummies(df_processed, columns=['uf'], prefix='uf', drop_first=True)
    
    # Definição das features
    base_features = [
        'media_cn', 'media_ch', 'media_lc', 'media_mt', 'media_redacao',
        'perc_privada', 'perc_urbana', 'perc_ingles', 'perc_redacoes_branco'
    ]
    engineered_features = ['log_pop_total', 'nota_x_proporcao_jovem']
    uf_features = [col for col in df_processed.columns if col.startswith('uf_')]
    
    features = base_features + engineered_features + uf_features
    target = 'vl_pib_per_capta'
    
    # Tratamento de Nulos
    df_processed.dropna(subset=features + [target], inplace=True)
    
    X = df_processed[features]
    y = np.log1p(df_processed[target])
    
    return X, y, df_processed.index

# =================================
# INÍCIO DO APP STREAMLIT
# =================================

with st.spinner('Carregando e processando dados... Por favor, aguarde.'):
    df = carregar_dados_agregados()

valores = {
    'media_cn': df['media_cn'].mean(), 'media_ch': df['media_cn'].mean(),
    'media_lc': df['media_lc'].mean(), 'media_mt': df['media_mt'].mean(),
    'media_redacao': df['media_redacao'].mean(), 'nota_media_geral': df['nota_media_geral'].mean()
}
df.fillna(value=valores, inplace=True)

st.sidebar.title('Navegação')
pagina_selecionada = st.sidebar.radio(
    "Ir para",
    ['1. Apresentação do Projeto', '2. Análise Exploratória', '3. Análise Preditiva e Relatório', '4. Conclusão']
)

# =================================
# PÁGINA 1: APRESENTAÇÃO DO PROJETO
# =================================
# =================================
# PÁGINA 1: APRESENTAÇÃO DO PROJETO
# =================================
if pagina_selecionada == '1. Apresentação do Projeto':
    st.title("Análise da Relação entre Desempenho no ENEM e Desenvolvimento Econômico Municipal")
    st.markdown("---")

    st.header('1. Problema de Pesquisa e Contextualização')
    st.markdown("""
    A educação é frequentemente citada como um pilar para o desenvolvimento socioeconômico. Este projeto investiga empiricamente essa premissa no contexto brasileiro, buscando responder à seguinte questão central:
    """)
    st.info("#### Há relação entre o desempenho dos estudantes do ensino médio e o crescimento econômico nos municípios brasileiros?")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Desempenho dos Estudantes")
        st.markdown("""
        O desempenho dos estudantes é uma métrica complexa, avaliada neste projeto com foco nos resultados do Exame Nacional do Ensino Médio (ENEM), mas que se conecta a um contexto mais amplo.
        
        - **Avaliação por Provas:** Neste estudo, o desempenho é medido pela **média das notas dos participantes do ENEM por município**. Outros indicadores importantes no cenário nacional incluem o Ideb e o Saeb.
        - **Fatores Socioeconômicos:** Estudos acadêmicos demonstram que o desempenho escolar está associado a indicadores como o IDH e a renda per capita, sugerindo que melhores condições de vida impulsionam a educação.
        """)

    with col2:
        st.subheader("Crescimento Econômico")
        st.markdown("""
        O crescimento econômico é abordado relacionando o desenvolvimento com a qualidade da educação, que funciona tanto como causa quanto consequência da prosperidade.
        
        - **PIB per capita:** A principal métrica utilizada para medir o crescimento econômico municipal neste e em outros estudos é o **Produto Interno Bruto (PIB) per capita**.
        - **Qualidade da Educação como Motor:** Há uma forte associação entre a qualidade educacional e taxas de crescimento. Um aumento na proficiência dos alunos está ligado a um aumento na taxa de crescimento do PIB.
        - **Geração de Empregos:** Municípios com melhor qualidade de ensino tendem a criar mais oportunidades de emprego para jovens, um indicador-chave de dinamismo econômico.
        """)
    
    st.markdown("---")
    st.header("2. Objetivo e Hipóteses")
    st.markdown("**Objetivo:** Identificar e explorar as relações existentes entre o desempenho dos estudantes do ensino médio e o crescimento econômico nos municípios brasileiros.")
    
    st.markdown("""
    Para guiar nossa análise, partimos das seguintes hipóteses:
    - **Hipótese 1:** Municípios com maior PIB per capita apresentam melhor desempenho dos estudantes do ensino médio.
    - **Hipótese 2:** Um aumento na qualidade da educação está positivamente correlacionado com o crescimento do PIB per capita municipal.
    """)


    st.markdown("---")
    st.header('3. Metodologia Analítica')
    st.write("A abordagem metodológica emprega três modelos de regressão com propósitos complementares, permitindo uma análise robusta tanto em termos de interpretabilidade quanto de capacidade preditiva.")
    
    # O restante da página 1 (colunas com os modelos) permanece o mesmo...
    col1_pag1, col2_pag2, col3_pag3 = st.columns(3)
    with col1_pag1:
        st.subheader('A. Regressão Linear Múltipla')
        st.write(r"""
            **Objetivo:** Interpretabilidade e quantificação de efeitos.
            Este modelo é utilizado para estimar a direção e a magnitude da relação linear entre as variáveis independentes e o PIB per capita.
            $$
            \log(PIB_{pc}) = \beta_0 + \beta_1 \cdot Nota_{ENEM} + \dots + \epsilon
            $$
        """)
    with col2_pag2:
        st.subheader("B. Árvore de Decisão")
        st.write("""
            **Objetivo:** Entendimento de regras e interações.
            A Árvore de Decisão segmenta os dados através de regras condicionais, criando um modelo visual e intuitivo para entender como as variáveis interagem.
        """)
    with col3_pag3:
        st.subheader("C. Random Forest")
        st.write("""
            **Objetivo:** Maximizar a acurácia preditiva.
            Este modelo cria múltiplas Árvores de Decisão e agrega seus resultados, capturando relações complexas e identificando as variáveis mais importantes.
        """)
    st.markdown("---")
    st.info("Navegue pelas seções no menu lateral para acessar a análise exploratória e os resultados dos modelos.")

# =================================
# PÁGINA 2: ANÁLISE EXPLORATÓRIA
# =================================
elif pagina_selecionada == "2. Análise Exploratória":
    # ... (O código desta página não precisa de alterações e permanece o mesmo)
    st.sidebar.header('Filtros para Análise')

    lista_ufs_original = sorted(df['uf'].dropna().unique())
    selecionar_todas = st.sidebar.checkbox('Selecionar Todas as UFs', value=True)
    ufs_padroes = lista_ufs_original if selecionar_todas else []
    ufs_selecionadas = st.sidebar.multiselect('Selecione a UF', options=lista_ufs_original, default=ufs_padroes)
    df_filtrado = df[df['uf'].isin(ufs_selecionadas)] if ufs_selecionadas else df
    
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Análise Geográfica", "📊 Relações e Correlações", "📈 Distribuições e Comparações", "🏆 Rankings Municipais"])
    with tab1:
        st.subheader('Distribuição Geográfica das Variáveis')
        variavel_mapa = st.selectbox("Selecione a variável para visualizar no mapa:",['vl_pib_per_capta', 'nota_media_geral'],index=None,format_func=lambda x: 'PIB per Capta' if x == 'vl_pib_per_capta' else 'Nota Média ENEM')
        if variavel_mapa:
            st.info("Passe o mouse sobre os municípios para ver os valores.")
            path_geojson = 'geojs-100-mun.json'
            
            @st.cache_data
            def load_geojson(path):
                with open(path, 'r', encoding='utf-8') as f:
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
                    feature['properties']['pib_formatado'] = f"R$ {dados.get('vl_pib_per_capta', 0):,.2f}"
                    feature['properties']['nota_formatada'] = f"{dados.get('nota_media_geral', 0):.2f}"
                else: 
                    feature['properties']['nome_municipio'] = 'Dado não disponível'
                    feature['properties']['uf'] = ''
                    feature['properties']['pib_formatado'] = 'N/A'
                    feature['properties']['nota_formatada'] = 'N/A'
            
            bins = list(df_filtrado[variavel_mapa].quantile(np.linspace(0, 1, 8)))
            mapa = folium.Map(location=[-14.2350, -51.9253], zoom_start=4)
            choropleth = folium.Choropleth(geo_data=geojson_data,data=df_filtrado,columns=['co_municipio_esc_str', variavel_mapa],key_on='feature.properties.id',fill_color='YlOrRd',fill_opacity=0.7,line_opacity=0.2,legend_name=f'Valor de {variavel_mapa}',bins=bins,highlight=True).add_to(mapa)
            tooltip = folium.GeoJsonTooltip(fields=['nome_municipio', 'uf', 'pib_formatado', 'nota_formatada'],aliases=['Município:', 'UF:', 'PIB per Capita:', 'Nota Média ENEM:'],sticky=True,style="background-color: #F0EFEF; border: 2px solid black; border-radius: 3px; box-shadow: 3px;")
            folium.GeoJson(geojson_data,style_function=lambda x: {'fillOpacity': 0, 'color': 'transparent', 'weight': 0}, tooltip=tooltip).add_to(mapa)
            folium_static(mapa, width=None, height=500)
    with tab2:
        st.subheader('Relação entre PIB per Capita e Nota Média no ENEM')
        st.write("Este gráfico de dispersão é o ponto central da nossa hipótese. Ele nos permite visualizar se existe uma tendência (positiva, negativa ou nula) entre o desempenho educacional e a riqueza municipal.")
        fig_scatter = px.scatter(df_filtrado,x='nota_media_geral',y='vl_pib_per_capta',hover_data=['nome_municipio', 'uf'],trendline='ols',trendline_color_override='red',log_y=True,labels={'nota_media_enem': 'Nota Média no ENEM','pib_per_capita': 'PIB per Capita (R$)'},title='PIB per Capita vs. Nota Média no ENEM')
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("O eixo Y (PIB per Capita) está em escala logarítmica para melhor visualização da relação, dado que alguns municípios possuem valores muito altos.")
        st.markdown('---')
        st.subheader('Mapa de Calor das Correlações')
        st.write("O mapa de calor quantifica a relação linear entre as principais variáveis numéricas. Valores próximos de 1 (azul escuro) indicam uma forte correlação positiva, enquanto valores próximos de -1 indicam uma forte correlação negativa. Valores próximos de 0 (cores claras) sugerem ausência de correlação linear.")
        cols_para_corr = ['media_cn', 'media_ch', 'media_lc', 'media_mt', 'media_redacao', 'nota_media_geral', 'vl_pib_per_capta', 'pop_total', 'proporcao_jovem']
        matriz_corr = df_filtrado[cols_para_corr].corr()
        rename_dict = {'media_cn': 'Ciências da Natureza','media_ch': 'Ciências Humanas','media_lc': 'Linguagens e Códigos','media_mt': 'Matemática','media_redacao': 'Redação','nota_media_geral': 'Nota Média Geral','vl_pib_per_capta': 'PIB per Capita','pop_total': 'População Total','proporcao_jovem': 'Proporção de Jovens'}
        matriz_corr_renamed = matriz_corr.rename(columns=rename_dict, index=rename_dict)
        fig_heatmap = px.imshow(matriz_corr_renamed,text_auto=True,aspect="auto",color_continuous_scale='RdBu_r', title="Correlação entre Variáveis Chave")
        st.plotly_chart(fig_heatmap, use_container_width=True)
    with tab3:
        st.subheader('Distribuição das Variáveis Chave')
        st.write("Os histogramas mostram a frequência dos valores para nossas principais variáveis, permitindo entender sua forma e dispersão.")
        col1, col2 = st.columns(2)
        with col1:
            fig_hist_enem = px.histogram(df_filtrado, x='nota_media_geral', nbins=50, title='Distribuição das Notas Médias do ENEM')
            st.plotly_chart(fig_hist_enem)
        with col2:
            fig_hist_pib = px.histogram(df_filtrado, x='vl_pib_per_capta', nbins=50, title='Distribuição do PIB per Capta')
            st.plotly_chart(fig_hist_pib)
        st.markdown('---')
        st.subheader('Comparações entre Unidades da Federação (UFs)')
        st.write("Os boxplots são ideais para comparar a distribuição de uma variável entre diferentes categorias. Aqui, podemos ver claramente as disparidades educacionais e econômicas entre os estados brasileiros.")
        variavel_boxplot = st.selectbox("Selecione a variável para comparar entre as UFs:",['nota_media_geral', 'vl_pib_per_capta'],format_func=lambda x: 'Nota Média ENEM' if x == 'nota_media_geral' else 'PIB per Capita')
        if variavel_boxplot and not df_filtrado.empty:
            ordem_medianas = df_filtrado.groupby('uf')[variavel_boxplot].median().sort_values(ascending=False).index
            fig_boxplot = px.box(df_filtrado,x='uf',y=variavel_boxplot,category_orders={'uf': ordem_medianas},title=f'Distribuição de {variavel_boxplot} por UF')
            st.plotly_chart(fig_boxplot, use_container_width=True)
    with tab4:
        st.subheader("Rankings Municipais")
        st.write("Analisar os extremos nos ajuda a entender os perfis dos municípios com maior e menor desempenho.")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Top 10 Municípios por Nota Média no ENEM")
            top_10_enem = df_filtrado.nlargest(10, 'nota_media_geral')
            fig_top_enem = px.bar(top_10_enem,x='nota_media_geral',y='nome_municipio',orientation='h',text='nota_media_geral',labels={'nome_municipio': 'Município', 'nota_media_geral': 'Nota Média'})
            fig_top_enem.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_top_enem.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_top_enem, use_container_width=True)
        with col2:
            st.markdown("#### Top 10 Municípios por PIB per Capita")
            top_10_pib = df_filtrado.nlargest(10, 'vl_pib_per_capta')
            fig_top_pib = px.bar(top_10_pib,x='vl_pib_per_capta',y='nome_municipio',orientation='h',text='vl_pib_per_capta',labels={'nome_municipio': 'Município', 'vl_pib_per_capta': 'PIB per Capita (R$)'})
            fig_top_pib.update_traces(texttemplate='R$ %{text:,.0f}', textposition='outside')
            fig_top_pib.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_top_pib, use_container_width=True)


# =======================================
# PÁGINA 3: ANÁLISE PREDITIVA E RELATÓRIO
# =======================================
elif pagina_selecionada == "3. Análise Preditiva e Relatório":
    st.header("Modelagem Preditiva: Prevendo o PIB per capita")
    st.write("""
        Nesta seção, treinamos três modelos de regressão para prever o PIB per capita
        de um município com base em suas características educacionais e demográficas.
        
        **Importante:** A variável PIB per capita possui uma distribuição muito assimétrica 
        (poucos municípios são muito ricos). Para melhorar o desempenho dos modelos, 
        aplicamos uma **transformação logarítmica** (log(1+x)) sobre ela. Além disso,
        as features numéricas foram **escalonadas (padronizadas)** para que tivessem média 0 e desvio padrão 1,
        uma prática que beneficia modelos como a Regressão Linear.
    """)

    # --- Pré-processamento e Divisão dos Dados ---
    X, y, _ = preprocessar_para_modelo(df)
    st.success(f"O modelo será treinado com {len(X)} municípios (após remover dados faltantes e aplicar transformações).")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Escalonamento das Features ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Treinamento dos modelos ---
    with st.spinner('Treinando modelos... Por favor, aguarde.'):
        # Modelo de Regressão Linear (com dados escalonados)
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        y_pred_lr = lr_model.predict(X_test_scaled)
        r2_lr = r2_score(y_test, y_pred_lr)
        rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

        # Modelo Random Forest (pode usar dados escalonados ou não, usamos por consistência)
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_scaled, y_train)
        y_pred_rf = rf_model.predict(X_test_scaled)
        r2_rf = r2_score(y_test, y_pred_rf)
        rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
        
        # Modelo Árvore de Decisão
        tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)
        tree_model.fit(X_train_scaled, y_train)
        y_pred_tree = tree_model.predict(X_test_scaled)
        r2_tree = r2_score(y_test, y_pred_tree)
        rmse_tree = np.sqrt(mean_squared_error(y_test, y_pred_tree))

    # --- Abas para visualização ---
    tab_comp, tab_lr, tab_rf, tab_tree = st.tabs(['🏆 Comparação', 'Regressão Linear', 'Random Forest', 'Árvore de Decisão'])

    with tab_comp:
        st.subheader("Comparação de Desempenho dos Modelos")
        
        df_results = pd.DataFrame({
            'Modelo': ['Random Forest', 'Regressão Linear', 'Árvore de Decisão'],
            'R² (R-quadrado)': [r2_rf, r2_lr, r2_tree],
            'RMSE (Erro Médio)': [rmse_rf, rmse_lr, rmse_tree]
        }).sort_values(by='R² (R-quadrado)', ascending=False)
        st.dataframe(df_results.set_index('Modelo').style.format('{:.3f}'))
        
        st.markdown("""
        **Interpretação:**
        - **R²:** Indica a porcentagem da variação no PIB per capita que o modelo consegue explicar. Mais perto de 1.0 é melhor.
        - **RMSE:** Mostra o erro médio das previsões na escala do log(PIB). Menor é melhor.
        
        O **Random Forest** apresentou o melhor desempenho preditivo (maior R² e menor RMSE), pois é capaz de capturar relações não-lineares complexas que a Regressão Linear não consegue.
        
        **Trade-off:** A **Regressão Linear** oferece maior interpretabilidade (seus coeficientes nos dizem o peso de cada variável), enquanto a **Árvore de Decisão** mostra regras de negócio claras, apesar de serem modelos geralmente menos precisos que o Random Forest.
        """)

    with tab_lr:
        st.header('Análise do Modelo de Regressão Linear')
        st.metric(label="R²", value=f"{r2_lr:.3f}")
        
        st.subheader("Interpretando os Coeficientes")
        st.info("Os coeficientes mostram o impacto de cada variável no log(PIB). Como as variáveis foram escalonadas, os coeficientes podem ser comparados entre si para ver qual tem mais 'peso' no modelo.")
        coefs = pd.DataFrame(lr_model.coef_, index=X.columns, columns=['Coeficiente'])
        st.dataframe(coefs.sort_values(by='Coeficiente', ascending=False).style.format('{:.4f}'))
        
        st.subheader('Análise de Resíduos')
        residuos = y_test - y_pred_lr
        df_residuos = pd.DataFrame({'Previsto': y_pred_lr, 'Resíduo': residuos})
        
        fig_res = px.scatter(df_residuos, x='Previsto', y='Resíduo', title='Resíduos vs. Valores Previstos')
        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_res, use_container_width=True)
        st.info("Idealmente, os pontos deveriam se distribuir aleatoriamente em torno da linha vermelha, sem formar padrões.")

    with tab_rf:
        st.header("Análise do Modelo Random Forest")
        st.metric(label="R²", value=f"{r2_rf:.3f}")
        
        st.subheader("Importância das Variáveis (Feature Importance)")
        st.write("Quais variáveis o modelo considerou mais importantes para fazer suas previsões? (Mostrando as 20 principais)")
        feature_importances = pd.DataFrame(rf_model.feature_importances_, index=X.columns, columns=['importance'])
        st.bar_chart(feature_importances.sort_values(by='importance', ascending=False).head(20))

    with tab_tree:
        st.header("Análise do Modelo de Árvore de Decisão")
        st.metric(label="R²", value=f"{r2_tree:.3f}")
        st.metric(label='RMSE', value=f"{rmse_tree:.3f}")
        
        st.subheader("Visualização da Árvore")
        st.write("Uma única árvore nos ajuda a entender as regras de decisão. (Limitada a 3 níveis para visualização)")
        fig, ax = plt.subplots(figsize=(25, 12))
        tree.plot_tree(tree_model, feature_names=X.columns, filled=True, rounded=True, fontsize=10, max_depth=3)
        st.pyplot(fig)
        
# =================================
# PÁGINA 4: CONCLUSÃO
# =================================
elif pagina_selecionada == "4. Conclusão":
    st.title('Relatório Analítico e Conclusão')
    st.header('Análise Interativa por Município')
    st.write('Selecione um município para ver seus dados e a predição do modelo Random Forest, que apresentou o melhor desempenho.')

    # --- Treinamento do Modelo Final com Pipeline ---
    with st.spinner('Treinando modelo final com todos os dados...'):
        X_full, y_full_log, df_final_index = preprocessar_para_modelo(df)
        
        # O pipeline garante que o escalonamento seja aplicado antes do treinamento
        pipeline_final = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
        ])
        
        pipeline_final.fit(X_full, y_full_log)
        
        # Fazendo predições em todos os dados que foram usados no treino
        predicoes_log = pipeline_final.predict(X_full)
    
    # Criamos um DataFrame de resultados com as colunas originais que queremos mostrar
    df_resultados = df.loc[df_final_index].copy()
    df_resultados['pib_predito'] = np.expm1(predicoes_log)
    df_resultados['diferenca_predicao'] = df_resultados['pib_predito'] - df_resultados['vl_pib_per_capta']
    df_resultados['display_name'] = df_resultados['nome_municipio'] + ' - ' + df_resultados['uf']

    municipio_selecionado = st.selectbox(
        "Selecione o Município",
        options=df_resultados['display_name'].sort_values(),
        index=None,
        placeholder="Digite o nome de um município..."
    )

    if municipio_selecionado:
        dados_municipio = df_resultados[df_resultados['display_name'] == municipio_selecionado].iloc[0]

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
                delta_color="off" # Cor neutra para a diferença
            )
        
        if dados_municipio['diferenca_predicao'] > 0:
            st.success(f"O modelo previu um PIB per capita **maior** que o real. Isso pode indicar que, com base em suas características educacionais e demográficas, o município tem um potencial econômico ainda não totalmente realizado.")
        else:
            st.warning(f"O modelo previu um PIB per capita **menor** que o real. Isso sugere que o município possui outros fatores de riqueza (ex: uma grande indústria, royalties de mineração, agronegócio intensivo) que não foram capturados pelas variáveis do modelo.")


        st.markdown("---")
        st.subheader("Dados Originais do Município")
        
        display_features = {
            'Nota Média Geral (ENEM)': f"{dados_municipio['nota_media_geral']:.2f}",
            'População Total': f"{dados_municipio['pop_total']:,.0f}",
            'Proporção de Jovens (15 a 19 anos)': f"{dados_municipio['proporcao_jovem']:.2%}"
        }
        st.table(pd.Series(display_features, name="Valor"))

    st.markdown("---")
    st.header('Análise das Hipóteses Iniciais')

    st.subheader("Hipótese 1: Municípios com maior PIB per capita apresentam melhor desempenho dos estudantes do ensino médio.")
    st.success("✔️ Veredito: Confirmada.")
    st.write("""
    A análise exploratória demonstrou uma clara e consistente correlação positiva entre o PIB per capita e a nota média geral no ENEM. 
    - O **gráfico de dispersão** (na página de Análise Exploratória) exibe uma linha de tendência ascendente, indicando que, em geral, municípios mais ricos têm estudantes com notas mais altas.
    - O **mapa de calor** quantifica essa relação com um coeficiente de correlação positivo.
    - Os **rankings** também mostram que municípios no topo de um indicador frequentemente aparecem bem posicionados no outro.
    """)

    st.subheader("Hipótese 2: Um aumento na qualidade da educação está positivamente correlacionado com o crescimento do PIB per capita municipal.")
    st.success("✔️ Veredito: Confirmada com forte evidência de correlação.")
    st.write("""
    Os resultados também suportam fortemente esta hipótese, mostrando uma associação positiva entre a qualidade da educação (proxy pela nota do ENEM) e a riqueza municipal.
    - A **análise preditiva** reforça essa ideia: as notas do ENEM nas diferentes áreas do conhecimento figuraram entre as variáveis mais importantes para o modelo Random Forest prever o PIB per capita de um município.
    - Isso indica que o desempenho educacional não é apenas um dado isolado, mas um forte preditor da realidade econômica local.
    """)
    st.info("""
    **Nota sobre Causalidade:** Embora a correlação seja forte e consistente em todas as análises, este estudo de corte transversal não pode afirmar a direção da causalidade. Ou seja, não podemos garantir se a educação de qualidade *causa* o aumento do PIB, ou se um PIB maior permite investimentos que *causam* uma melhoria na educação. O mais provável é que ambos se retroalimentem em um ciclo virtuoso.
    """)