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

def draw_graph(rules, rules_to_show):
  import networkx as nx  
  G1 = nx.DiGraph()
   
  color_map=[]
  N = 50
  colors = np.random.rand(N)    
  strs=['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11']   
   
  for i in range (rules_to_show):      
    G1.add_nodes_from(["R"+str(i)])
    
    for a in rules.iloc[i]['antecedants']:
                
        G1.add_nodes_from([a])
        
        G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)
       
    for c in rules.iloc[i]['consequents']:
             
            G1.add_nodes_from()
            
            G1.add_edge("R"+str(i), c, color=colors[i],  weight=2)
 
  for node in G1:
       found_a_string = False
       for item in strs: 
           if node==item:
                found_a_string = True
       if found_a_string:
            color_map.append('yellow')
       else:
            color_map.append('green')    

  edges = G1.edges()
  colors = [G1[u][v]['color'] for u,v in edges]
  weights = [G1[u][v]['weight'] for u,v in edges]
 
  pos = nx.spring_layout(G1, k=16, scale=1)
  nx.draw(G1, pos, edges=edges, node_color = color_map, edge_color=colors, width=weights, font_size=16, with_labels=False)            
   
  for p in pos:  # raise text positions
           pos[p][1] += 0.07
  nx.draw_networkx_labels(G1, pos)
  plt.show()

st.title('Envie um arquivo')
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:

    print("LENDO OS DADOS")

    bytes_data = uploaded_file.getvalue()
    df = pd.read_csv(uploaded_file, sep=',')

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
        #unique_columns.append('Cliente')
        #unique_columns.append('Cód Rep.')
        unique_columns.append('Rep')
        unique_columns.append('Categoria agrupada')
        df = df[unique_columns]
        df = df[ df['Mês-ano'] != '1/5/2023' ]

        df = df.rename( {ds_colname:'ds', y_colname:'y'} , axis=1)
        df['ds'] = pd.to_datetime(df['ds'], format='%d/%m/%Y')
        #df['Cód Rep.'] = df['Cód Rep.'].astype(str)
        df['Rep'] = df['Rep'].astype(str)
        #df['Cliente'] = df['Cliente'].astype(str)
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
        #['Top', 'Cód Rep.', 'Categoria agrupada'],
        ['Top', 'Rep', 'Categoria agrupada'],
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

        Y_test_df = Y_df.groupby('unique_id').tail(5)
        Y_train_df = Y_df.drop(Y_test_df.index)

        Y_test_df = Y_test_df.set_index('unique_id')
        Y_train_df = Y_train_df.set_index('unique_id')

        #print(Y_train_df)

        fcst = StatsForecast(df=Y_train_df,
                            models = [HistoricAverage(),
                            Naive(),
                            SeasonalNaive(season_length = 12)],
                            freq = 'MS', n_jobs=1)
                            
        Y_hat_df = fcst.forecast(h=12, fitted=True)
        Y_fitted_df = fcst.forecast_fitted_values()
        reconcilers = [BottomUp()]
        hrec = HierarchicalReconciliation(reconcilers=reconcilers)
        Y_rec_df = hrec.reconcile(Y_hat_df=Y_hat_df, Y_df=Y_fitted_df, S=S_df, tags=tags)
        #st.write(Y_rec_df.head(5))
        Y_train_df.reset_index(inplace=True)
        errors = {}
        errors['ids'] = []
        errors['erro'] = []

        df_merged = Y_test_df.merge(Y_rec_df.reset_index(), on=['unique_id', 'ds'])
        Y_rec_df2 = Y_rec_df.reset_index()
    
        for id in df_merged['unique_id'].unique():
            fig = plt.figure(figsize=(10, 2))
            erro = mean_absolute_percentage_error( df_merged.loc[ df_merged['unique_id'] == id, 'y' ], df_merged.loc[ df_merged['unique_id'] == id, 'SeasonalNaive' ] )
            #print(id, erro )
            if erro < 2.0 and (Y_train_df.loc[ Y_train_df['unique_id'] == id, 'y'] == 0).sum() < len(Y_train_df.loc[ Y_train_df['unique_id'] == id, 'y'])*0.2:
                plt.plot(  Y_train_df.loc[ Y_train_df['unique_id'] == id, 'ds'],  Y_train_df.loc[ Y_train_df['unique_id'] == id, 'y'], label='treino' )
                plt.plot(df_merged.loc[ df_merged['unique_id'] == id, 'ds' ],  df_merged.loc[ df_merged['unique_id'] == id, 'y' ], label='real')
                plt.plot(df_merged.loc[ df_merged['unique_id'] == id, 'ds' ],  df_merged.loc[ df_merged['unique_id'] == id, 'SeasonalNaive' ], label='Seasonal')
                plt.plot(Y_rec_df2.loc[ Y_rec_df2['unique_id'] == id, 'ds' ],  Y_rec_df2.loc[ Y_rec_df2['unique_id'] == id, 'SeasonalNaive' ], label='Previsão')
                n = len(Y_rec_df2.loc[ Y_rec_df2['unique_id'] == id, 'ds' ])
                plt.plot( Y_rec_df2.loc[ Y_rec_df2['unique_id'] == id, 'ds' ],  [Y_train_df.loc[ Y_train_df['unique_id'] == id, 'y'].mean()]*n, label='Meta' )
                plt.legend()
                plt.title(id + ' ' + str(round( mean_absolute_percentage_error( df_merged.loc[ df_merged['unique_id'] == id, 'y' ], df_merged.loc[ df_merged['unique_id'] == id, 'SeasonalNaive' ] ) ,2)))
                plt.xticks(rotation = 45)
                #plt.show()
                st.pyplot(fig)

        rules_categoria = pd.read_pickle('regras_recomendacoes.pkl')
        best_rules = rules_categoria.sort_values(by='support', ascending=False)
        best_rules = best_rules[['antecedents', 'consequents', 'support']]
        best_rules.columns = ['Item Vendido', 'Recomendação', 'Frequência da Regra']

        item = []
        item_rec = []

        for i, row in best_rules.iterrows():
            str1 = ''
            str2 = ''
            for x in row['Item Vendido']:
                str1 += x + ' '
            for y in row['Recomendação']:
                str2 += y + ' '
            item.append(str1)
            item_rec.append(str2)
        best_rules['Item Vendido'] = item
        best_rules['Recomendação'] = item_rec



        st.write('## Recomendações mais comuns:')
        st.write(best_rules)
        df = df[ df['ds'] >= Y_hat_df['ds'].min() ]
        df = add_rec(df, rules_categoria)
        csv = convert_df(df)
        st.download_button(
            "Baixar previsões",
            csv,
            "file.csv",
            "text/csv",
            key='download-csv'
        )
        
