import pandas as pd
import psycopg2 as pg
import streamlit as st
import time
import warnings
from warnings import filterwarnings

filterwarnings("ignore", category=UserWarning,
               message=".*pandas only supports SQLAlchemy connectable.*")

db_credenciais = st.secrets['postgresql']

host = db_credenciais['host']
port = db_credenciais['port']
user = db_credenciais['user']
password = db_credenciais['password']
dbname = db_credenciais['dbname']

with pg.connect(host=host, port=port, user=user, password=password, dbname=dbname) as conn:
    curr = conn.cursor()

    inicio = time.time()

    enem_2024 = pd.read_sql_query("""SELECT
                                            co_municipio_esc,
                                            nota_cn_ciencias_da_natureza as nota_cn,
                                            nota_ch_ciencias_humanas as nota_ch,
                                            nota_lc_linguagens_e_codigos as nota_lc,
                                            nota_mt_matematica as nota_mt,
                                            nota_redacao
                                    FROM
                                            public.ed_enem_2024_resultados
                                            """, conn)

    pib_municipios = pd.read_sql_query("""SELECT
                                                codigo_municipio_dv,
                                                ano_pib,
                                                vl_pib,
                                                vl_pib_per_capta
                                        FROM
                                                public.pib_municipios
                                                """, conn)

    censo = pd.read_sql_query("""SELECT
                                    "CO_MUNICIPIO",
                                    "TOTAL",
                                    "IDADE"
                                FROM
                                    public."Censo_20222_Populacao_Idade_Sexo"
                                WHERE
                                    "IDADE" BETWEEN 15 AND 19
                                    """, conn)

    municipio = pd.read_sql_query("""SELECT
                                            nome_municipio,
                                            codigo_municipio_dv
                                    FROM
                                            public.municipio
                                            """, conn)

    uf = pd.read_sql_query("""SELECT
                                cd_uf,
                                sigla_uf,
                                nome_uf
                            FROM
                                public.unidade_federacao
                        """, conn)

    df_temp = pd.merge(enem_2024, pib_municipios, left_on='co_municipio_esc',
                       right_on='codigo_municipio_dv', how='left')

    df = pd.merge(df_temp, censo, left_on='co_municipio_esc',
                  right_on='CO_MUNICIPIO', how='left')

    fim = time.time()
    print((fim - inicio) * 1000)
    print(df.shape)
