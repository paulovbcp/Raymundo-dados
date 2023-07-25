import streamlit as st
import pandas as pd
from io import StringIO
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.graph_objs as go
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

#if 'loaded_data' not in st.session_state:
#    st.session_state['loaded_date'] = 1
st.title('Envie um arquivo')
uploaded_file = st.file_uploader("Choose a file")
trained_model = False

if uploaded_file is not None:


    print("LENDO OS DADOS")

    bytes_data = uploaded_file.getvalue()
    df = pd.read_csv(uploaded_file, sep=',')

    if len(df) == 0:
        st.write('DataFrame vazio!')

    if len(df) > 0:
        st.write('## Dados enviados')
        st.write( df.head(5) )
    
    print(df.columns)
    ds_colname = st.selectbox(
    'Escolha a coluna da data?',
    (df.columns))

    y_colname = st.selectbox(
    'Escolha a coluna do valor a ser previsto',
    (df.columns))

    hierarquias = []
    hierarquia = st.multiselect(
    'Escolha a primeira hierarquia:',
    df.columns)
    if len(hierarquia) > 0:
        hierarquias.append(hierarquia)

    hierarquia = st.multiselect(
    'Escolha a segunda hierarquia:',
    df.columns)
    if len(hierarquia) > 0:
        hierarquias.append(hierarquia)

    hierarquia = st.multiselect(
    'Escolha a terceira hierarquia:',
    df.columns)
    if len(hierarquia) > 0:
        hierarquias.append(hierarquia)

    teste = st.number_input('Escolha quantos pontos quer usar para teste', step=1)
    previsao = st.number_input('Escolha quantos pontos quer usar para previsão', step=1)

    unique_columns = []
    for hierarquia in hierarquias:
        for h in hierarquia:
            if h not in unique_columns:
                unique_columns.append(h)

    un_col_lite = unique_columns

    if st.button('Treinar modelo'):

        unique_columns.append(y_colname)
        unique_columns.append(ds_colname)
        df = df[unique_columns]
        df = df[ df['Mês-ano'] != '1/5/2023' ]

        df = df.rename( {ds_colname:'ds', y_colname:'y'} , axis=1)
        df['ds'] = pd.to_datetime(df['ds'], format='%d/%m/%Y')
        df['Top'] = 'Total'

        # esses filtros tão muito hardcoded pra essa aplicação. Remover
        
        for col in un_col_lite:
            if col in df.columns:
                if 'SP' in df[col].unique():
                    col_estado = col
        
        if col_estado:
            df = df[df[col_estado].isin(df[col_estado].value_counts().index[ df[col_estado].value_counts() > 3000 ].tolist())]
        
        df = df.dropna()
# =============================================================================
#         df = df[df['ESTADO'].isin(df['ESTADO'].value_counts().index[ df['ESTADO'].value_counts() > 3000 ].tolist())]
#         if 'Categoria agrupada' in df.columns:
#             df = df[ ~df['Categoria agrupada'].isin(['ÁLCOOL', 'Clorados', 'Outros', 'Cloro Gel']) ]
#         if 'Canal_novo+velho' in df.columns:
#             df = df[ ~df['Canal_novo+velho'].isin(['Inativo', 'Terceirização', '(vazio)']) ]
#         
# =============================================================================

        spec = [
        ['Top'],
        ]
        for hierarquia in hierarquias:
            hierarquia.insert(0, 'Top')
            spec.append(hierarquia)

        print(spec)
        Y_df, S_df, tags = aggregate(df, spec)
        Y_df = Y_df.reset_index()

        Y_test_df = Y_df.groupby('unique_id').tail(teste)
        Y_train_df = Y_df.drop(Y_test_df.index)

        Y_test_df = Y_test_df.set_index('unique_id')
        Y_train_df = Y_train_df.set_index('unique_id')

        fcst = StatsForecast(df=Y_train_df,
                            models = [#AutoARIMA(),
                            Naive(),
                            SeasonalNaive(season_length = 12)],
                            freq = 'MS', n_jobs=-1)
        Y_hat_df = fcst.forecast(h=teste+previsao, fitted=True)
        Y_fitted_df = fcst.forecast_fitted_values()
        reconcilers = [BottomUp()]
        hrec = HierarchicalReconciliation(reconcilers=reconcilers)
        Y_rec_df = hrec.reconcile(Y_hat_df=Y_hat_df, Y_df=Y_fitted_df, S=S_df, tags=tags)
        Y_rec_df.reset_index(inplace=True)
        Y_train_df.reset_index(inplace=True)
        errors = {}
        errors['ids'] = []
        errors['erro'] = []

        df_merged = Y_test_df.merge(Y_rec_df.reset_index(), on=['unique_id', 'ds'])

        csv = convert_df(df)
        st.download_button(
            "Baixar previsões",
            csv,
            "file.csv",
            "text/csv",
            key='download-csv'
        )
        trained_model = True
        #for id in df_merged['unique_id'].unique():
    if trained_model:
        print('a')
        #serie_plot = st.selectbox(
        #    'Qual serie deseja visualizar?',
        #    df_merged['unique_id'].unique())

        #if serie_plot in df_merged['unique_id'].unique():
        #    fig = go.Figure([
        #        go.Scatter(
        #            name='Measurement',
        #            x=Y_train_df.loc[ Y_train_df['unique_id'] == id, 'ds'],
        #            y=Y_train_df.loc[ Y_train_df['unique_id'] == id, 'y'],
        #            mode='lines',
        #            line=dict(color='rgb(31, 119, 180)'),
        #        ),
                #go.Scatter(
                #    name='Upper Bound',
                #    x=df['Time'],
                #    y=df['10 Min Sampled Avg']+df['10 Min Std Dev'],
                #    mode='lines',
                #    marker=dict(color="#444"),
                #    line=dict(width=0),
                #    showlegend=False
                #),
                #go.Scatter(
                #    name='Lower Bound',
                #    x=df['Time'],
                #    y=df['10 Min Sampled Avg']-df['10 Min Std Dev'],
                #    marker=dict(color="#444"),
                #    line=dict(width=0),
                #    mode='lines',
                #    fillcolor='rgba(68, 68, 68, 0.3)',
                #    fill='tonexty',
                #    showlegend=False
                #)
        #    ])
        #    fig.update_layout(
        #        yaxis_title='Wind speed (m/s)',
        #        title='Continuous, variable value error bars',
        #        hovermode="x"
        #    )
        #    fig.show()
        fig = go.Figure()
        for id in df_merged['unique_id'].unique():
            #fig = plt.figure(figsize=(10, 2))
                #print(id, mean_absolute_percentage_error( df_merged.loc[ df_merged['unique_id'] == id, 'y' ], df_merged.loc[ df_merged['unique_id'] == id, 'SeasonalNaive' ] ) )
            #plt.plot(  Y_train_df.loc[ Y_train_df['unique_id'] == id, 'ds'],  Y_train_df.loc[ Y_train_df['unique_id'] == id, 'y'], label='treino' )
            #plt.plot(df_merged.loc[ df_merged['unique_id'] == id, 'ds' ],  df_merged.loc[ df_merged['unique_id'] == id, 'y' ], label='real')
            #plt.plot(df_merged.loc[ df_merged['unique_id'] == id, 'ds' ],  df_merged.loc[ df_merged['unique_id'] == id, 'SeasonalNaive' ], label='Seasonal')
            
            #print(Y_rec_df.head(2))
            #plt.plot(  Y_rec_df.loc[ Y_rec_df['unique_id'] == id, 'ds' ],  Y_rec_df.loc[ Y_rec_df['unique_id'] == id, 'SeasonalNaive' ], label='Previsão' )
            #plt.legend()
            #plt.title(id + ' ' + str(round( mean_absolute_percentage_error( df_merged.loc[ df_merged['unique_id'] == id, 'y' ], df_merged.loc[ df_merged['unique_id'] == id, 'SeasonalNaive' ] ) ,2)))
            #plt.xticks(rotation = 45)
            #plt.show()
            #st.pyplot(fig)

            tmp_train = Y_train_df[ Y_train_df['unique_id'] == id ]
            tmp_merged = df_merged[ df_merged['unique_id'] == id ]
            tmp_pred = Y_rec_df[ Y_rec_df['unique_id'] == id ]
            tmp_pred = tmp_pred[ tmp_pred['ds']  > tmp_merged['ds'].max() ]
            print(tmp_pred.head(2))
            fig.add_trace(go.Scatter(
                    x=tmp_train['ds'],
                    y=tmp_train['y'],
                    mode='lines',
                    name=id+'_treino'
                ))
            fig.add_trace(go.Scatter(
                    x=tmp_merged['ds'],
                    y=tmp_merged['y'],
                    mode='lines',
                    name=id+'_teste'
                ))
            fig.add_trace(go.Scatter(
                    x=tmp_merged['ds'],
                    y=tmp_merged['SeasonalNaive'],
                    mode='lines',
                    name=id+'_teste_previsão'
                ))
            fig.add_trace(go.Scatter(
                    x=tmp_pred['ds'],
                    y=tmp_pred['SeasonalNaive'],
                    mode='lines',
                    name=id+'_previsão'
                ))
            

        st.plotly_chart(fig, use_container_width=True)

        