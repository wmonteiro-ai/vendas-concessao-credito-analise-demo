import asyncio
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

##########
#Imports
##########
import streamlit as st
import pandas as pd
import numpy as np
import pickle

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from scipy.stats import kendalltau

import seaborn as sns
import matplotlib.pyplot as plt

##########
#Configuração do Streamlit
##########
st.set_page_config(page_title='Inteledge - Análise de Crédito', page_icon="💡", layout="centered", initial_sidebar_state="auto", menu_items=None)

sns.set_context("talk")

##########
#Funções auxiliares
##########
@st.cache(allow_output_mutation=True)
def get_samples(target):
    # carregando uma amostra da base de dados
    df = pickle.load(open('df_resampled.pkl', 'rb'))
    columns = [target] + df.drop(target, axis=1).columns.tolist()
	
    return df

def calculate_corr(df, method='kendall'):
    if method != 'kendall':
        df = df.select_dtypes('number')
        return df.corr(method=method)
        
    correlations = []

    for col in df.columns:
        correlation_row = []
        for row in df.columns:
            correlation_row.append(kendalltau(df[row], df[col]).correlation)

        correlations.append(correlation_row)

    return pd.DataFrame(correlations, columns=df.columns, index=df.columns).fillna(0)

##########
#Preparando o simulador
##########
# carregando uma amostra da base de dados
target = 'Aprovar o crédito?'
df = get_samples(target)

##########
#Seção 1 - Cabeçalho
##########
col1, _, _ = st.columns(3)
with col1:
    st.image('inteledge.png')

st.title('Analytics para Dados de Crédito')
st.markdown('O nosso trabalho não se resume a apenas criar algoritmos de Inteligência Artificial ou de gerar gráficos bonitos por si só, mas também o de gerar *insights* para você. Abaixo você pode ver alguns exemplos do tipo de conhecimento que podemos trazer ao analisar os dados.')
st.markdown('Entenda os gráficos abaixo como exemplos, somente: existem várias possibilidades de análise de dados e determinamos as técnicas que fazem mais sentido especificamente para a base de dados que você possui. Ficou interessado? Entre em contato conosco e nos siga em [@inteledge.lab](https://instagram.com/inteledge.lab) no Instagram!')
st.markdown('Também [veja o simulador que construímos com IA para esta base de dados](https://share.streamlit.io/wmonteiro92/vendas-concessao-credito-xai-demo/main/predictions_xai.py).')

# Amostra
st.header('Amostra da base de dados')
st.dataframe(df.replace({target: {1: 'Aprovado', 0: 'Reprovado'}}).sample(100))

##########
#Seção 2 - Correlações
##########
st.header('Correlações')
st.write('Existem várias colunas: algumas delas são numéricas (como idade e valores) e outras são categóricas. E, na vida real, nem sempre as correlações são perfeitamente lineares e envolvem números. Para isso, utilizamos a correlação de Kendall para encontrar esses relacionamentos entre categorias, grupos e valores. Interpretamos assim:')
"""
* Valores acima de `+0.30`: correlação forte - quando um aumenta, o outro aumenta;
* Valores abaixo de `-0.30`: correlação forte - quando um aumenta, o outro diminui.
"""

df_corr = calculate_corr(df, method='kendall').round(2)
df_corr = df_corr.replace({1: np.nan})
cols = df_corr.rename(columns={'Tempo morando na residência atual': 'Tempo na resid. atual', 'Número de empréstimos passados': 'Nº empréstimos passados'}).columns.tolist()

fig = px.imshow(df_corr, labels=dict(color="Correlação"),
                x=cols,
                y=cols,
                text_auto=True
               )
fig.update_xaxes(side="top")
fig.update_layout(width=600, height=600)
st.plotly_chart(fig)

st.write('Passe o mouse (ou toque, se estiver no celular) nos quadrados. Veja que existem alguns deles com uma cor diferente dos demais. Vamos analisar algumas dessas combinações de uma forma mais aprofundada agora.')

##########
#Seção 3 - Investigando correlações
##########
st.header('Valor pedido x Motivo do empréstimo')
st.markdown('No gráfico anterior vimos que há uma alta correlação entre o `Valor pedido` e o `Motivo do empréstimo`. Veja a diferença entre os diferentes motivos de empréstimo abaixo: os valores de empréstimo para uma casa são bem maiores do que aqueles para móveis, por exemplo.')
st.markdown('Você pode interagir com o gráfico abaixo à vontade ao ligar e desligar diferentes motivos de empréstimo. Pode também dar zoom e passar o mouse (ou tocar, caso esteja com um celular) nos diferentes motivos para ver mais detalhes. Observe também os riscos acima dos gráficos: cada risco representa um cliente, e podemos ver de outra maneira a distribuição entre eles. Legal, não é?')

# Histograma
fig = px.histogram(df.replace({target: {1: 'Aprovado', 0: 'Reprovado'}}),
                   x='Valor pedido', color='Motivo do empréstimo',
                   marginal='rug', facet_col=target)
fig.update_layout(xaxis_title='Número de clientes')
st.plotly_chart(fig)

st.header('Visão multidimensional')
st.markdown('Também conseguimos visualizar, ao mesmo tempo, a relação de diferentes dados. Como exemplo, temos aqui algumas categorias que tiveram alta correlação com a aprovação do crédito (ou não). Veja que realmente existem categorias em que predominam mais aprovações do que reprovações, e vice-versa. Isso nos ajuda a entender melhor os relacionamentos desses dados. Fique à vontade para interagir com o gráfico.')
# Coordenadas paralelas
dimensions = []
for col in ['Saldo na conta', 'Histórico de empréstimos', 'É estrangeiro?', 'Motivo do empréstimo']:
    dimensions.append(go.parcats.Dimension(
        values=df[col].values,
        label=col
    ))

dimensions.append(go.parcats.Dimension(
    values=df['Aprovar o crédito?'].values,
    label='Aprovar o crédito?',
    categoryarray=[0, 1],
    ticktext=['Reprovado', 'Aprovado']
))

colorscale = [[0, '#92017D'], [1, '#0167CD']];
fig = go.Figure(data=[go.Parcats(dimensions=dimensions,
                                 line={'color': df['Aprovar o crédito?'],
                                       'colorscale': colorscale},
                                 hoveron='color', hoverinfo='count+probability')])
st.plotly_chart(fig)

# Violin plot
st.markdown('Também conseguimos ver o relacionamento dos dados numéricos. Aqui, por exemplo, vemos a distribuição entre número de parcelas x aprovação (ou não) do crédito. Perceba que há uma *distribuição* maior de reprovações quanto maior é o número de parcelas.')
fig = px.violin(df.replace({target: {0: 'Reprovado', 1: 'Aprovado'}}),
          y='Número de parcelas',
          x=target,
          box=True,
          color=target)
fig.update_layout(showlegend=False)
st.plotly_chart(fig)

# Pairplot
st.markdown('Se quiser, também podemos ver vários desses relacionamentos ao mesmo tempo com várias colunas. Veja que nem todos os dados são bem divididos: não há uma linha divisória entre o que é aprovação e o que é reprovação, e tudo está bem misturado. Isto pode se assemelhar muito com os seus próprios dados: mesmo assim, conseguimos criar bons modelos de IA para prever resultados.')
st.pyplot(sns.pairplot(data=df, plot_kws={'alpha': 0.1}, hue=target))

# Distribuição dos dados categóricos
st.header('Dados categóricos')
st.markdown('Também conseguimos ver facilmente a *distribuição* de diferentes dados categóricos. Pessoas com casa própria podem ter o seu crédito aprovado de forma mais fácil do que pessoas que somente tem um carro (ou nenhum bem), por exemplo.')
st.write('Note também algumas coisas interessantes: desempregados possuem uma chance maior de terem o seu empréstimo aprovado do que os demais grupos - isto pode levantar uma questão interessante: será que temos poucos dados deste grupo? Ou este grupo possui outras características (como alguns bens) que facilitaram a tomada de decisão? Ou é algum erro nos dados? É este o tipo de pergunta e insight que queremos construir em conjunto.')

for col in df.select_dtypes('object').columns[-8:]:
    values_aprovado = pd.DataFrame(df[df[target]==1].groupby(col).count()[target]).reset_index(drop=False).values.T
    values_reprovado = pd.DataFrame(df[df[target]==0].groupby(col).count()[target]).reset_index(drop=False).values.T
    
    fig = go.Figure()
    fig.add_bar(x=values_aprovado[0], y=100*values_aprovado[1]/(values_aprovado[1]+values_reprovado[1]), name='Aprovado')
    fig.add_bar(x=values_reprovado[0], y=100*values_reprovado[1]/(values_aprovado[1]+values_reprovado[1]), name='Reprovado')
    fig.update_layout(barmode="relative", title=f'Distribuição dos dados para "{col}"')
    st.plotly_chart(fig)
    
st.markdown('Siga-nos no Instagram! [@inteledge.lab](https://instagram.com/inteledge.lab)')