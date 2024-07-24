import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
import hashlib

# Configuração da página para modo wide
st.set_page_config(layout="wide")

# Carregar os dados
@st.cache_data
def load_data():
    df = pd.read_csv('Credit_Card_MarceloCorni.csv')
    df.columns = [
        'Unnamed','ID','LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 
        'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 
        'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 
        'PAY_AMT5', 'PAY_AMT6', 'default.payment.next.month'
    ]
    return df

# Carregar o texto de descrição
@st.cache_data
def load_description():
    with open('creditcard.txt', encoding='utf-8') as f:
        content = f.read()
    return content

df = load_data()

# Removendo a coluna Unnamed
df = df.drop(columns=['Unnamed'])

# Renomeando colunas para facilitar a análise
df.rename(columns={'default.payment.next.month': 'DEFAULT_PAYMENT', 'PAY_0': 'PAY_1'}, inplace=True)

# Adicionando novos atributos
df['AVG_PAY'] = df[['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].mean(axis=1)
df['AVG_BILL_AMT'] = df[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].mean(axis=1)
df['AVG_PAY_AMT'] = df[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].mean(axis=1)
df['TOTAL_BILL_AMT'] = df[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].sum(axis=1)
df['TOTAL_PAY_AMT'] = df[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].sum(axis=1)

description_content = load_description()

# Título do App
st.title('Análise de Inadimplência de Cartões de Crédito em Taiwan')

# Menu lateral
st.sidebar.title('Menu')
menu_options = ['Apresentação do Trabalho','Descrição dos Dados', 'Análise e Tratamento de Dados', 'Análise Gráfica', 'Outliers e Normalização', 'Conclusões']
selection = st.sidebar.radio('Ir para', menu_options)

# Função para gerar o URL do Gravatar a partir do e-mail
def get_gravatar_url(email, size=100):
    # Criar o hash MD5 do e-mail
    email_hash = hashlib.md5(email.strip().lower().encode('utf-8')).hexdigest()
    # Construir a URL do Gravatar
    gravatar_url = f"https://www.gravatar.com/avatar/{email_hash}?s={size}"
    return gravatar_url

# Definir o e-mail e o tamanho da imagem
email = "marcelo@desenvolvedor.net"  # Substitua pelo seu e-mail
size = 200  # Tamanho da imagem

# Obter o URL do Gravatar
gravatar_url = get_gravatar_url(email, size)


if selection == 'Apresentação do Trabalho':
    # Layout principal com colunas
    col1, col2 = st.columns([1, 3])

    # Conteúdo da coluna esquerda
    with col1:
        st.image(gravatar_url, caption="", use_column_width=True)


    # Conteúdo da coluna direita
    with col2:
        st.write("## Marcelo Corni Alves")
        st.write("Julho/2024")
        st.write("Disciplina: Mineração de Dados")


elif selection == 'Descrição dos Dados':
    st.header('Descrição dos Dados')
    st.text(description_content)

elif selection == 'Análise e Tratamento de Dados':
    st.header('Análise e Tratamento de dados')    
    st.write(df.describe())
    st.text('Número de linhas: {}'.format(df.shape[0]))
    st.subheader('Tratamentos nas colunas')
    st.write('» Coluna default.payment.next.month foi renomeada para DEFAULT_PAYMENT')
    st.write('» Coluna que estava sem header no arquivo foi removida')
    st.write('» Coluna PAY_1 foi renomeada para manter o padrão dos demais campos')
    st.subheader('Substituição de valores faltantes e tratamento de inconsistências.')
    st.write('» Colunas de PAY_1 a PAY_6 foram preenchidas os valores nulos ou na com -2 (sem consumo) e convertido o tipo para int')
    st.write('» Colunas BILL_AMT1 a BILL_AMT6 e PAY_AMT1 a PAY_AMT6 foram preenchidas os valores nulos ou na com 0')  
    st.write('» Coluna MARRIAGE foi agrupada os valores que dizem a mesma coisa em um único valor')
    st.write('» Coluna EDUCATION foi agrupada os valores que dizem a mesma coisa em um único valor')    

    st.subheader('Colunas com valores nulos antes dos tratamentos')
    nulldata = df.isnull().sum()
    nulldata.rename('Quantidade de valores nulos', inplace=True)
    st.write(nulldata)

    # tratamento das colunas PAY_1 a PAY_6 para preencher os valores nulos ou NA com 0
    for i in range(1, 7):
        df['PAY_' + str(i)] = df['PAY_' + str(i)].fillna(0.0)
        
    # tratamento das colunas BILL_AMT1 a BILL_AMT6 e PAY_AMT1 a PAY_AMT6 para preencher os valores nulos ou NA com 0
    for i in range(1, 7):
        df['BILL_AMT' + str(i)] = df['BILL_AMT' + str(i)].fillna(0.0)
        df['PAY_AMT' + str(i)] = df['PAY_AMT' + str(i)].fillna(0.0)
        df['PAY_AMT' + str(i)] = df['PAY_AMT' + str(i)].astype(int)

    # tratamento da coluna MARRIAGE agrupar os valores que dizem a mesma coisa em um único valor
    df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})
    df['MARRIAGE'] = df['MARRIAGE'].astype(int)

    # tratamento da coluna EDUCATION agrupar os valores que dizem a mesma coisa em um único valor
    df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})

    df['AVG_PAY'] = df['AVG_PAY'].fillna(0.0)
    df['AVG_PAY_AMT'] = df['AVG_PAY_AMT'].fillna(0.0)
    
    df['DEFAULT_PAYMENT'] = df['DEFAULT_PAYMENT'].fillna(0.0)
    df['DEFAULT_PAYMENT'] = df['DEFAULT_PAYMENT'].astype(int)

    st.subheader('Colunas com valores nulos depois dos tratamentos')
    nulldata = df.isnull().sum()
    nulldata.rename('Quantidade de valores nulos', inplace=True)
    st.write(nulldata)

    st.subheader('Remoção de registros duplicados')
    total = df.shape[0]
    df.drop_duplicates(inplace=True)
    total_apos_remocao_duplicados = total - df.shape[0]
    st.write('Foram removidos {} registros duplicados'.format(total_apos_remocao_duplicados))

    st.subheader('Verificação dos valores seguindo orientações do creditcard.txt')
    
    st.write('Valores únicos da coluna SEX')
    unique_sex = df['SEX'].unique()
    st.write(unique_sex)
    
    st.write('Valores únicos da coluna EDUCATION')
    unique_education = df['EDUCATION'].unique()    
    st.write(unique_education)
    
    st.write('Valores únicos da coluna MARRIAGE')    
    unique_marriage = df['MARRIAGE'].unique()
    st.write(unique_marriage)
    
    unique_pay = [df['PAY_1'].unique(),df['PAY_2'].unique(),df['PAY_3'].unique(),df['PAY_4'].unique(),df['PAY_5'].unique(),df['PAY_6'].unique()]
    st.write('Valores únicos das colunas PAY_1 a PAY_6')

    unique_default = df['DEFAULT_PAYMENT'].unique()
    col1,col2,col3,col4,col5,col6 = st.columns(6)

    with col1:
        st.write('PAY_1')
        st.write(unique_pay[0])
    with col2:
        st.write('PAY_2')
        st.write(unique_pay[1])
    with col3:
        st.write('PAY_3')
        st.write(unique_pay[2])
    with col4:
        st.write('PAY_4')
        st.write(unique_pay[3])
    with col5:
        st.write('PAY_5')
        st.write(unique_pay[4])
    with col6:
        st.write('PAY_6')
        st.write(unique_pay[5])
    
    st.write('Valores únicos da coluna DEFAULT_PAYMENT')
    st.write(unique_default)

elif selection == 'Análise Gráfica':
    st.header('Análise Gráfica')
    
    # Filtros
    sex_filter = st.sidebar.selectbox('Sexo', options=[0, 1, 2], format_func=lambda x: 'Todos' if x == 0 else ('Masculino' if x == 1 else 'Feminino'))
    education_filter = st.sidebar.multiselect('Educação', options=[1, 2, 3, 4], default=[1, 2, 3, 4], format_func=lambda x: ['Pós-graduação', 'Universidade', 'Ensino Médio', 'Outros'][x-1])
    marriage_filter = st.sidebar.multiselect('Estado Civil', options=[1, 2, 3], default=[1, 2, 3], format_func=lambda x: ['Casado', 'Solteiro', 'Outros'][x-1])
    
    limit_bal_filter = st.sidebar.slider('Limite de Crédito', float(df['LIMIT_BAL'].min()), float(df['LIMIT_BAL'].max()), (float(df['LIMIT_BAL'].min()), float(df['LIMIT_BAL'].max())))
    age_filter = st.sidebar.slider('Idade', int(df['AGE'].min()), int(df['AGE'].max()), (int(df['AGE'].min()), int(df['AGE'].max())))

    pay_filters = {
        'PAY_1': st.sidebar.multiselect('Histórico de Pagamento - Setembro', options=list(range(-2, 10)), default=list(range(-2, 10))),
        'PAY_2': st.sidebar.multiselect('Histórico de Pagamento - Agosto', options=list(range(-2, 10)), default=list(range(-2, 10))),
        'PAY_3': st.sidebar.multiselect('Histórico de Pagamento - Julho', options=list(range(-2, 10)), default=list(range(-2, 10))),
        'PAY_4': st.sidebar.multiselect('Histórico de Pagamento - Junho', options=list(range(-2, 10)), default=list(range(-2, 10))),
        'PAY_5': st.sidebar.multiselect('Histórico de Pagamento - Maio', options=list(range(-2, 10)), default=list(range(-2, 10))),
        'PAY_6': st.sidebar.multiselect('Histórico de Pagamento - Abril', options=list(range(-2, 10)), default=list(range(-2, 10)))
    }
    
    bill_amt_filters = {
        'BILL_AMT1': st.sidebar.slider('Fatura - Setembro', float(df['BILL_AMT1'].min()), float(df['BILL_AMT1'].max()), (float(df['BILL_AMT1'].min()), float(df['BILL_AMT1'].max()))),
        'BILL_AMT2': st.sidebar.slider('Fatura - Agosto', float(df['BILL_AMT2'].min()), float(df['BILL_AMT2'].max()), (float(df['BILL_AMT2'].min()), float(df['BILL_AMT2'].max()))),
        'BILL_AMT3': st.sidebar.slider('Fatura - Julho', float(df['BILL_AMT3'].min()), float(df['BILL_AMT3'].max()), (float(df['BILL_AMT3'].min()), float(df['BILL_AMT3'].max()))),
        'BILL_AMT4': st.sidebar.slider('Fatura - Junho', float(df['BILL_AMT4'].min()), float(df['BILL_AMT4'].max()), (float(df['BILL_AMT4'].min()), float(df['BILL_AMT4'].max()))),
        'BILL_AMT5': st.sidebar.slider('Fatura - Maio', float(df['BILL_AMT5'].min()), float(df['BILL_AMT5'].max()), (float(df['BILL_AMT5'].min()), float(df['BILL_AMT5'].max()))),
        'BILL_AMT6': st.sidebar.slider('Fatura - Abril', float(df['BILL_AMT6'].min()), float(df['BILL_AMT6'].max()), (float(df['BILL_AMT6'].min()), float(df['BILL_AMT6'].max())))
    }
    
    filtered_df = df.copy()
    if sex_filter != 0:
        filtered_df = filtered_df[filtered_df['SEX'] == sex_filter]
    filtered_df = filtered_df[filtered_df['EDUCATION'].isin(education_filter)]
    filtered_df = filtered_df[filtered_df['MARRIAGE'].isin(marriage_filter)]
    filtered_df = filtered_df[(filtered_df['LIMIT_BAL'] >= limit_bal_filter[0]) & (filtered_df['LIMIT_BAL'] <= limit_bal_filter[1])]
    filtered_df = filtered_df[(filtered_df['AGE'] >= age_filter[0]) & (filtered_df['AGE'] <= age_filter[1])]
    
    for col, values in pay_filters.items():
        filtered_df = filtered_df[filtered_df[col].isin(values)]
    
    for col, (min_val, max_val) in bill_amt_filters.items():
        filtered_df = filtered_df[(filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)]
    
    fig1 = px.histogram(filtered_df, x='LIMIT_BAL', title='Distribuição do Limite de Crédito')
    fig1.update_traces(marker=dict(line=dict(color='black', width=1)))
    fig2 = px.box(filtered_df, x='SEX', y='LIMIT_BAL', title='Limite de Crédito por Sexo')
    fig2.update_traces(marker=dict(line=dict(color='black', width=1)))
    fig3 = px.scatter(filtered_df, x='AGE', y='LIMIT_BAL', color='DEFAULT_PAYMENT', title='Limite de Crédito vs Idade')
    fig3.update_traces(marker=dict(line=dict(color='black', width=1)))
    fig4 = px.histogram(filtered_df, x='AVG_PAY', color='DEFAULT_PAYMENT', title='Distribuição da Média de Atrasos')
    fig4.update_traces(marker=dict(line=dict(color='black', width=1)))
    fig5 = px.histogram(filtered_df, x='AVG_BILL_AMT', color='DEFAULT_PAYMENT', title='Distribuição da Média das Faturas')
    fig5.update_traces(marker=dict(line=dict(color='black', width=1)))
    fig6 = px.histogram(filtered_df, x='AVG_PAY_AMT', color='DEFAULT_PAYMENT', title='Distribuição da Média dos Pagamentos')
    fig6.update_traces(marker=dict(line=dict(color='black', width=1)))

    st.plotly_chart(fig1)
    st.plotly_chart(fig2)
    st.plotly_chart(fig3)
    st.plotly_chart(fig4)
    st.plotly_chart(fig5)
    st.plotly_chart(fig6)

    # Matriz de Correlação
    # Selecionar apenas as colunas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Calcular a matriz de correlação
    corr_matrix = df[numeric_cols].corr()

    # Plotar a matriz de correlação usando Plotly
    fig_corr = px.imshow(corr_matrix, 
                        text_auto=True, 
                        template='plotly',
                        color_continuous_scale='RdBu_r', 
                        title='Matriz de Correlação',
                        width=1920, 
                        height=1080)

    st.plotly_chart(fig_corr, use_container_width=True)

elif selection == 'Outliers e Normalização':
    st.header('Outliers e Normalização')

    # Detecção de Outliers
    st.write('Detecção e remoção de outliers utilizando várias técnicas.')

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:

        st.subheader('Colunas Numéricas')
        st.write(numeric_cols)

        # Preencher valores ausentes com a mediana
        df_filled = df[numeric_cols].apply(lambda x: x.fillna(x.median()))

        # IQR
        st.subheader('Método IQR (Interquartile Range)')
        st.write('O método IQR utiliza o intervalo interquartil (IQR), que é a diferença entre o terceiro quartil (75º percentil) e o primeiro quartil (25º percentil). Valores fora do intervalo [Q1 - 1.5*IQR, Q3 + 1.5*IQR] são considerados outliers.')
        q1 = df_filled.quantile(0.25)
        q3 = df_filled.quantile(0.75)
        iqr = q3 - q1
        df_iqr = df_filled[~((df_filled < (q1 - 1.5 * iqr)) | (df_filled > (q3 + 1.5 * iqr))).any(axis=1)]

        fig_iqr1 = px.box(df, y='LIMIT_BAL', title='Boxplot do Limite de Crédito (IQR)')
        fig_iqr2 = px.box(df_iqr, y='LIMIT_BAL', title='Boxplot do Limite de Crédito Após Remoção de Outliers (IQR)')
        st.plotly_chart(fig_iqr1)
        if not df_iqr.empty:
            st.plotly_chart(fig_iqr2)
        else:
            st.markdown('<span style="color:red">Todos os dados foram removidos como outliers pelo método IQR.</span>', unsafe_allow_html=True)

        # Z-Score
        st.subheader('Método Z-Score')
        st.write('O método Z-Score calcula a distância de cada ponto em relação à média em termos de desvios padrão. Valores com um Z-Score maior que 3 ou menor que -3 são considerados outliers.')
        from scipy import stats
        z_scores = np.abs(stats.zscore(df_filled))
        threshold = 3
        df_zscore = df_filled[(z_scores < threshold).all(axis=1)]
        st.write('Z-Scores calculados:')
        st.write(z_scores)

        fig_zscore1 = px.box(df, y='LIMIT_BAL', title='Boxplot do Limite de Crédito (Z-Score)')
        fig_zscore2 = px.box(df_zscore, y='LIMIT_BAL', title='Boxplot do Limite de Crédito Após Remoção de Outliers (Z-Score)')
        st.plotly_chart(fig_zscore1)
        if not df_zscore.empty:
            st.plotly_chart(fig_zscore2)
        else:
            st.markdown('<span style="color:red">Todos os dados foram removidos como outliers pelo método Z-Score.</span>', unsafe_allow_html=True)

        # Modified Z-Score
        st.subheader('Método Modified Z-Score')
        st.write('O método Modified Z-Score é uma variação do Z-Score que utiliza a mediana e a mediana da diferença absoluta (MAD). É mais robusto para distribuições não normais.')
        def modified_z_score(series):
            median = np.median(series)
            mad = np.median(np.abs(series - median))
            return 0.6745 * (series - median) / mad

        threshold = 3.5
        mod_z_scores = df_filled.apply(modified_z_score)
        df_mod_zscore = df_filled[(np.abs(mod_z_scores) < threshold).all(axis=1)]
        st.write('Modified Z-Scores calculados:')
        st.write(mod_z_scores)

        fig_mod_zscore1 = px.box(df, y='LIMIT_BAL', title='Boxplot do Limite de Crédito (Modified Z-Score)')
        fig_mod_zscore2 = px.box(df_mod_zscore, y='LIMIT_BAL', title='Boxplot do Limite de Crédito Após Remoção de Outliers (Modified Z-Score)')
        st.plotly_chart(fig_mod_zscore1)
        if not df_mod_zscore.empty:
            st.plotly_chart(fig_mod_zscore2)
        else:
            st.markdown('<span style="color:red">Todos os dados foram removidos como outliers pelo método Modified Z-Score.</span>', unsafe_allow_html=True)

        # Isolation Forest
        st.subheader('Método Isolation Forest')
        st.write('O método Isolation Forest isola observações ao construir árvores de decisão aleatórias. É eficaz para conjuntos de dados de alta dimensão.')
        from sklearn.ensemble import IsolationForest
        iso_forest = IsolationForest(contamination=0.1)
        y_pred = iso_forest.fit_predict(df_filled)
        df_iso_forest = df_filled[y_pred != -1]

        fig_iso_forest1 = px.box(df, y='LIMIT_BAL', title='Boxplot do Limite de Crédito (Isolation Forest)')
        fig_iso_forest2 = px.box(df_iso_forest, y='LIMIT_BAL', title='Boxplot do Limite de Crédito Após Remoção de Outliers (Isolation Forest)')
        st.plotly_chart(fig_iso_forest1)
        if not df_iso_forest.empty:
            st.plotly_chart(fig_iso_forest2)
        else:
            st.markdown('<span style="color:red">Todos os dados foram removidos como outliers pelo método Isolation Forest.</span>', unsafe_allow_html=True)

        # DBSCAN
        st.subheader('Método DBSCAN')
        st.write('O método DBSCAN (Density-Based Spatial Clustering of Applications with Noise) é um algoritmo de clustering que pode identificar outliers como pontos que não pertencem a nenhum cluster denso.')
        from sklearn.cluster import DBSCAN
        dbscan = DBSCAN(eps=3, min_samples=2)
        clusters = dbscan.fit_predict(df_filled)
        df_dbscan = df_filled[clusters != -1]

        fig_dbscan1 = px.box(df, y='LIMIT_BAL', title='Boxplot do Limite de Crédito (DBSCAN)')
        fig_dbscan2 = px.box(df_dbscan, y='LIMIT_BAL', title='Boxplot do Limite de Crédito Após Remoção de Outliers (DBSCAN)')
        st.plotly_chart(fig_dbscan1)
        if not df_dbscan.empty:
            st.plotly_chart(fig_dbscan2)
        else:
            st.markdown('<span style="color:red">Todos os dados foram removidos como outliers pelo método DBSCAN.</span>', unsafe_allow_html=True)

        # Seletor de técnica de normalização
        st.subheader('Normalização dos Dados')
        normalization_technique = st.selectbox('Escolha a técnica de normalização:', ['Min-Max Scaling', 'Z-Score Normalization', 'Robust Scaler', 'MaxAbs Scaler', 'Log Transformation'])

        if normalization_technique == 'Min-Max Scaling':
            st.write('A normalização Min-Max reescala os valores dos dados para um intervalo específico, geralmente [0, 1].')
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            df_normalized = pd.DataFrame(scaler.fit_transform(df_filled), columns=numeric_cols)
        
        elif normalization_technique == 'Z-Score Normalization':
            st.write('A normalização Z-Score transforma os dados para que tenham média 0 e desvio padrão 1.')
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            df_normalized = pd.DataFrame(scaler.fit_transform(df_filled), columns=numeric_cols)

        elif normalization_technique == 'Robust Scaler':
            st.write('A normalização Robust Scaler utiliza a mediana e o intervalo interquartil (IQR) para normalizar os dados, sendo robusta contra outliers.')
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            df_normalized = pd.DataFrame(scaler.fit_transform(df_filled), columns=numeric_cols)

        elif normalization_technique == 'MaxAbs Scaler':
            st.write('A normalização MaxAbs Scaler escala os dados pelo valor absoluto máximo.')
            from sklearn.preprocessing import MaxAbsScaler
            scaler = MaxAbsScaler()
            df_normalized = pd.DataFrame(scaler.fit_transform(df_filled), columns=numeric_cols)

        elif normalization_technique == 'Log Transformation':
            st.write('A transformação logarítmica aplica a função logarítmica aos dados para reduzir a variação e tornar a distribuição mais próxima da normal.')
            df_normalized = df_filled.apply(np.log1p)

        st.write('Dados após a normalização:')
        st.write(df_normalized.head())
        st.write('Quantidade de registros após a normalização: {}'.format(df_normalized.shape[0]))

    else:
        st.write('Nenhuma coluna numérica encontrada para detecção de outliers e normalização.')


elif selection == 'Conclusões':
    # Adicionar coluna de inadimplência
    df['INADIMPLENTE'] = df['DEFAULT_PAYMENT']

    st.header('Conclusões')

    st.write('''
    1. **Média de Atrasos de Pagamento (AVG_PAY):**
        - Clientes com uma média de atrasos mais alta tendem a ter uma maior probabilidade de inadimplência.''')
    
    fig1 = px.histogram(df, x='AVG_PAY', color='INADIMPLENTE', nbins=20, title='Média de Atrasos de Pagamento (AVG_PAY)')
    fig1.update_layout(barmode='overlay')
    fig1.update_traces(opacity=0.75)
    fig1.update_traces(marker=dict(line=dict(color='black', width=1)))
    st.plotly_chart(fig1)

    st.write('''
    2. **Média das Faturas (AVG_BILL_AMT):**
        - Clientes com faturas médias mais altas também mostram uma tendência maior para inadimplência, especialmente quando combinado com altos atrasos médios.''')
    
    fig2 = px.histogram(df, x='AVG_BILL_AMT', color='INADIMPLENTE', nbins=20, title='Média das Faturas (AVG_BILL_AMT)')
    fig2.update_layout(barmode='overlay')
    fig2.update_traces(opacity=0.75)
    fig2.update_traces(marker=dict(line=dict(color='black', width=1)))
    st.plotly_chart(fig2)

    st.write('''
    3. **Média dos Pagamentos (AVG_PAY_AMT):**
        - Clientes que pagam valores médios mais baixos em relação às suas faturas tendem a ser mais propensos a inadimplência.''')

    fig3 = px.histogram(df, x='AVG_PAY_AMT', color='INADIMPLENTE', nbins=20, title='Média dos Pagamentos (AVG_PAY_AMT)')
    fig3.update_layout(barmode='overlay')
    fig3.update_traces(opacity=0.75)
    fig3.update_traces(marker=dict(line=dict(color='black', width=1)))
    st.plotly_chart(fig3)

    st.write('''
    4. **Limite de Crédito e Idade:**
        - Embora não seja um indicador isolado forte, combinações de alto limite de crédito com idades extremas (muito jovens ou mais velhos) podem também indicar uma maior propensão para inadimplência.''')
    
    fig4 = px.scatter(df, x='LIMIT_BAL', y='AGE', color='INADIMPLENTE', title='Limite de Crédito e Idade')
    fig4.update_traces(marker=dict(line=dict(color='black', width=1)))
    st.plotly_chart(fig4)

    st.write('''
    5. **Educação e Estado Civil:**
        - Certos níveis de educação e estado civil mostram correlações com inadimplência, com pessoas solteiras e com níveis de educação mais baixos apresentando maior risco.''')

    fig5 = px.histogram(df, x='EDUCATION', color='INADIMPLENTE', nbins=5, title='Educação e Inadimplência')
    fig5.update_traces(marker=dict(line=dict(color='black', width=1)))
    st.plotly_chart(fig5)
    
    fig6 = px.histogram(df, x='MARRIAGE', color='INADIMPLENTE', nbins=3, title='Estado Civil e Inadimplência')
    fig6.update_traces(marker=dict(line=dict(color='black', width=1)))
    st.plotly_chart(fig6)

    st.write('''
    **Recomendações:**
    - A análise sugere que instituições financeiras devem monitorar mais de perto os clientes com altos atrasos médios e faturas altas, oferecendo suporte e intervenções proativas para mitigar o risco de inadimplência.
    - Implementar políticas de crédito mais restritivas para clientes com histórico de pagamentos problemáticos e considerar esses insights na modelagem de risco de crédito.
    ''')

