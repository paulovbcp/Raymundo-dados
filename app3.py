import streamlit as st
import pandas as pd
from io import StringIO
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsforecast import StatsForecast
from statsforecast.models import HistoricAverage, Naive, SeasonalNaive, RandomWalkWithDrift, WindowAverage, AutoARIMA
from hierarchicalforecast.utils import aggregate
from sklearn.metrics import mean_absolute_percentage_error
from hierarchicalforecast.methods import BottomUp
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.evaluation import HierarchicalEvaluation
from hierarchicalforecast.utils import aggregate

@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

def add_rec(df_prev, rules_categoria):
    recs = {}
    recs['recomendação'] = []
    recs['categoria'] = []
    for categoria in df_prev['Categoria agrupada'].unique():
        rec_items = []
        for conjunto in rules_categoria.loc[ (rules_categoria['antecedents'] == {categoria}) & (rules_categoria['confidence'] > 0.7) ].sort_values('confidence', ascending=False).loc[:, 'consequents']:
            for item in conjunto:
                rec_items.append(item)
        rec_items = list(set(rec_items))
        recs['recomendação'].append(rec_items)
        recs['categoria'].append(categoria)
    prev_recs = pd.DataFrame.from_dict(recs)

    return df_prev.merge(prev_recs, left_on = 'Categoria agrupada', right_on='categoria'  )

st.title('Envie um arquivo')
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:

    print("LENDO OS DADOS")

    bytes_data = uploaded_file.getvalue()
    df = pd.read_csv(uploaded_file, sep=';')

    if len(df) == 0:
        st.write('DataFrame vazio!')

    if 'Mês-ano' not in df.columns:
        st.write('A planilha precisa de uma coluna chamada Mês-ano com as datas')
    
    if 'VOLUME' not in df.columns:
        st.write('A planilha precisa de uma coluna chamada VOLUME com o volume vendido')

    if len(df) > 0:
        st.write('## Dados enviados')
        st.write( df.head(5) )

    ds_colname = 'Mês-ano'
    y_colname = 'VOLUME'
    print(df.columns)
    #ds_colname = st.selectbox(
    #'Escolha a coluna da data?',
    #(df.columns))

    #y_colname = st.selectbox(
    #'Escolha a coluna do valor a ser previsto',
    #(df.columns))

    #hierarquias = []
    #hierarquia = st.multiselect(
    #'Escolha a primeira hierarquia:',
    #df.columns)
    #if len(hierarquia) > 0:
    #    hierarquias.append(hierarquia)

    #hierarquia = st.multiselect(
    #'Escolha a segunda hierarquia:',
    #df.columns)
    #if len(hierarquia) > 0:
    #    hierarquias.append(hierarquia)

    #hierarquia = st.multiselect(
    #'Escolha a terceira hierarquia:',
    #df.columns)
    #if len(hierarquia) > 0:
    #    hierarquias.append(hierarquia)

    unique_columns = []
    #for hierarquia in hierarquias:
    #    for h in hierarquia:
    #        if h not in unique_columns:
    #            unique_columns.append(h)



    if st.button('Treinar modelo'):

        unique_columns.append(y_colname)
        unique_columns.append(ds_colname)
        unique_columns.append('Cliente')
        unique_columns.append('Cód Rep.')
        unique_columns.append('Categoria agrupada')
        df = df[unique_columns]
        df = df[ df['Mês-ano'] != '1/5/2023' ]

        df = df.rename( {ds_colname:'ds', y_colname:'y'} , axis=1)
        df['ds'] = pd.to_datetime(df['ds'], format='%d/%m/%Y')
        df['Cód Rep.'] = df['Cód Rep.'].astype(str)
        df['Top'] = 'Total'

        # esses filtros tão muito hardcoded pra essa aplicação. Remover
        #df = df[df['UF'].isin(df['UF'].value_counts().index[ df['UF'].value_counts() > 3000 ].tolist())]
        #if 'Categoria agrupada' in df.columns:
        #    df = df[ ~df['Categoria agrupada'].isin(['ÁLCOOL', 'Clorados', 'Outros', 'Cloro Gel']) ]
        #if 'CANAL VENDA' in df.columns:
        #    df = df[ ~df['CANAL VENDA'].isin(['Inativo', 'Terceirização', '(vazio)']) ]
        df = df.dropna()

        spec = [
        ['Top'],
        #['Top', 'UF'],
        ['Top', 'Cód Rep.', 'Categoria agrupada'],
        #['Top', 'Cliente'],
        #['Top', 'UF', 'Categoria agrupada'  ],
        #['Top', 'CANAL VENDA', 'Categoria agrupada'],
        #['Top', 'UF', 'Categoria agrupada', 'CANAL VENDA'],
        ]
        #for hierarquia in hierarquias:
        #    hierarquia.insert(0, 'Top')
        #    spec.append(hierarquia)
        hierarquias = spec
        #print(spec)
        Y_df, S_df, tags = aggregate(df, spec)
        Y_df = Y_df.reset_index()

        Y_test_df = Y_df.groupby('unique_id').tail(15)
        Y_train_df = Y_df.drop(Y_test_df.index)

        Y_test_df = Y_test_df.set_index('unique_id')
        Y_train_df = Y_train_df.set_index('unique_id')

        fcst = StatsForecast(df=Y_train_df,
                            models = [HistoricAverage(),
                            Naive(),
                            SeasonalNaive(season_length = 12)],
                            freq = 'MS', n_jobs=-1)
        Y_hat_df = fcst.forecast(h=15, fitted=True)
        Y_fitted_df = fcst.forecast_fitted_values()
        reconcilers = [BottomUp()]
        hrec = HierarchicalReconciliation(reconcilers=reconcilers)
        Y_rec_df = hrec.reconcile(Y_hat_df=Y_hat_df, Y_df=Y_fitted_df, S=S_df, tags=tags)
        Y_train_df.reset_index(inplace=True)
        errors = {}
        errors['ids'] = []
        errors['erro'] = []

        df_merged = Y_test_df.merge(Y_rec_df.reset_index(), on=['unique_id', 'ds'])
        
        for id in df_merged['unique_id'].unique():
            fig = plt.figure(figsize=(10, 2))
            print(id, mean_absolute_percentage_error( df_merged.loc[ df_merged['unique_id'] == id, 'y' ], df_merged.loc[ df_merged['unique_id'] == id, 'SeasonalNaive' ] ) )
            plt.plot(  Y_train_df.loc[ Y_train_df['unique_id'] == id, 'ds'],  Y_train_df.loc[ Y_train_df['unique_id'] == id, 'y'], label='treino' )
            plt.plot(df_merged.loc[ df_merged['unique_id'] == id, 'ds' ],  df_merged.loc[ df_merged['unique_id'] == id, 'y' ], label='real')
            plt.plot(df_merged.loc[ df_merged['unique_id'] == id, 'ds' ],  df_merged.loc[ df_merged['unique_id'] == id, 'SeasonalNaive' ], label='Seasonal')
            plt.legend()
            plt.title(id + ' ' + str(round( mean_absolute_percentage_error( df_merged.loc[ df_merged['unique_id'] == id, 'y' ], df_merged.loc[ df_merged['unique_id'] == id, 'SeasonalNaive' ] ) ,2)))
            plt.xticks(rotation = 45)
            #plt.show()
            st.pyplot(fig)

        rules_categoria = pd.read_pickle('regras_recomendacoes.pkl')
        df = add_rec(df, rules_categoria)
        csv = convert_df(df)
        st.download_button(
            "Baixar previsões",
            csv,
            "file.csv",
            "text/csv",
            key='download-csv'
        )
        
