import pandas as pd
import numpy as np
import streamlit as st
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import datetime
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
from PIL import Image
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pickle import load

st.set_page_config(
    page_title="Defesa Greici Capellari",
    page_icon="random",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
                 'About': "# Contato: greicicapellari@gmail.com"
                 }
)

image = Image.open('diabetes.jfif')
st.sidebar.image(image)

#título
st.markdown("<h1 style='text-align: center; color: black;'>Aplicativo web desenvolvido para defesa de tese de doutorado.</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black;'>Greici Capellari</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: black;'>Criado por Lincoln Moura e Greici Capellari</h4>", unsafe_allow_html=True)

with st.expander("Emails para contato: "):
    st.write("lincolnsobral@yahoo.com.br - [https://www.linkedin.com/in/lincoln-moura-115323124](linkedin.com/in/lincoln-moura-115323124)")
    st.write("greicicapellari@gmail.com - [https://www.linkedin.com/in/greici-capellari-fabrizzio-51a310154](linkedin.com/in/greici-capellari-fabrizzio-51a310154)")


st.info("**Esta aplicação de inteligência artificial tem como objetivo fornecer uma ferramenta de "
        "análise preditiva** para auxílio a tomada de decisão dos profissionais. No lado esquerdo da tela,"
        " insira as variáveis referente as informações clínicas do paciente e verifique o resultado do banco de dados da COVID."
        )

#st.error("Ferramenta de análise criada para demonstração.")

#dataset
df = pd.read_excel("banco_limpo.xlsx", index_col=0)


#nomedousuário
st.sidebar.markdown("<h3 style='text-align: center; color: black;'>Análise do banco de dados</h3>", unsafe_allow_html=True)


hospital = st.sidebar.multiselect(
     'Escolha uma ou mais opções de banco de dados que deseja visualizar os gráficos:',
     ['hospital_UFAM', 'hospital_UFSC', 'hospital_UNIFESP', 'hospital_UFRN', 'hospital_UFRJ', 'Todos'],
     [])

if 'Todos' in  hospital:
    hospital = ['hospital_UFAM', 'hospital_UFSC', 'hospital_UNIFESP', 'hospital_UFRN', 'hospital_UFRJ']

if hospital!=[]:
        df2 = pd.DataFrame()
        for i in hospital:
            print(i)
            frames = [df2,df[df[i]==1]]
            df2 = pd.concat(frames)
        df2 = df2.reset_index(drop=True)
        df2
        
        col1, col2, col3 = st.columns(3)
        fig1 = px.histogram(df2, x="dias_uti", color="dias_uti", width=450, height=450)
        fig1.update_layout(bargap=0.2)
        col1.plotly_chart(fig1)
    
        fig2 = px.histogram(df2, x="idade", width=450, height=450)
        fig2.update_layout(bargap=0.2)
        col2.plotly_chart(fig2)
    
        fig = px.histogram(df2, x="pessoas_domicilio", width=450, height=450)
        fig.update_layout(bargap=0.2)
        col3.plotly_chart(fig)
    
        
        for var in [3,7,11,15]:                    
            col1, col2, col3, col4 = st.columns(4)
            coluna = [col1, col2, col3, col4]
            count=0
            for col in df2.iloc[:,var:var+4].columns:
                print(col)
                fig = px.pie(df2, values=df2[col], names=df2.dias_uti, title=str(col), width=300, height=300)
                coluna[count].plotly_chart(fig)            
                count+=1

st.sidebar.markdown("<h3 style='text-align: center; color: black;'> Esta opção habilita a entrada das variáveis para o modelo de predição.</h3>", unsafe_allow_html=True)

age = st.sidebar.selectbox(
        'Selecione a Idade:',
        (18, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,89, 91, 92, 96, 97, 99))

pessoas_domicilio = st.sidebar.selectbox(
        'Selecione a Idade:',
        (1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 12, 20))

hospital = st.sidebar.selectbox(
 'Escolha o hospital de origem do paciente:',
 ('hospital_UFAM', 'hospital_UFSC', 'hospital_UNIFESP', 'hospital_UFRN', 'hospital_UFRJ'))

sexo = st.sidebar.selectbox(
     'Preencha o sexo do paciente:',
     ('Masc', 'Fem', 'outro'))

instrucao = st.sidebar.selectbox(
     'Preencha a instrucao do paciente:',
     ('sem_inst', 'fund_inc', 'fund_com', 'med_inc', 'med_comp', 'sup_inc', 'sup_com'))

raca = st.sidebar.selectbox(
     'Preencha a raca do paciente:',
     ('branca', 'preta', 'parda', 'indigena', 'amarela', 'sem_resp'))

renda = st.sidebar.selectbox(
     'Preencha a renda do paciente:',
     ('ate_2mil', '2_5_mil', '5_10_mil', 'maior_10mil', 'sem_rend'))

fuma = st.sidebar.selectbox(
     'Preencha se paciente fumante:',
     ('nao_fumante', 'fumante', 'ex_fumante'))

tuple_outros = ('doenca_respiratoria_cronica',
       'hipertensa', 'doencas_cardiovasculares', 'diabetes', 'doencas_renais',
       'obesidade', 'cancer', 'febre', 'fadiga', 'dispneia', 'tosse',
       'perda_olfato_paladar', 'cefaleia', 'dor_corpo', 'nausea_vomito',
       'diarreia')

outros = st.sidebar.multiselect    (
     'Selecione um ou mais sintomas em caso da existencia:',
      tuple_outros)

def get_user_date():
    output = df.iloc[0,:]
    
    output.idade = age
    output.pessoas_domicilio = pessoas_domicilio
    
    for i in ['hospital_UFAM', 'hospital_UFSC', 'hospital_UNIFESP', 'hospital_UFRN', 'hospital_UFRJ']:
        if i in hospital:
            output[i] = 1
        else:
            output[i] = 0
            
    for i in ['Masc', 'Fem', 'outro']:
        if i in sexo:
            output[i] = 1
        else:
            output[i] = 0
            
    for i in ['sem_inst', 'fund_inc', 'fund_com', 'med_inc', 'med_comp', 'sup_inc', 'sup_com']:
        if i in instrucao:
            output[i] = 1
        else:
            output[i] = 0
            
    for i in ['branca', 'preta', 'parda', 'indigena', 'amarela', 'sem_resp']:
        if i in raca:
            output[i] = 1
        else:
            output[i] = 0
            
    for i in ['ate_2mil', '2_5_mil', '5_10_mil', 'maior_10mil', 'sem_rend']:
        if i in renda:
            output[i] = 1
        else:
            output[i] = 0
    
    for i in ['nao_fumante', 'fumante', 'ex_fumante']:
        if i == fuma:
            output[i] = 1
        else:
            output[i] = 0
            
    for i in ['doenca_respiratoria_cronica',
           'hipertensa', 'doencas_cardiovasculares', 'diabetes', 'doencas_renais',
           'obesidade', 'cancer', 'febre', 'fadiga', 'dispneia', 'tosse',
           'perda_olfato_paladar', 'cefaleia', 'dor_corpo', 'nausea_vomito',
           'diarreia']:
        if i in outros:
            output[i] = 1
        else:
            output[i] = 0
            
    return  pd.DataFrame(output).T

if st.sidebar.button('Realizar a análise'):
    #coletando as informações para predição
    user_input_variables = get_user_date()
    user_input_variables = user_input_variables.drop(['dias_uti'],axis=1)
    
    #https://machinelearningmastery.com/how-to-save-and-load-models-and-data-preparation-in-scikit-learn-for-later-use/    
    model = load(open('model.pkl', 'rb'))
    scaler = load(open('scaler.pkl', 'rb'))
    
    user_input_variables[['idade','pessoas_domicilio']] = scaler.transform(user_input_variables[['idade','pessoas_domicilio']])
    yhat = model.predict(user_input_variables)
    yhat_prob = model.predict_proba(user_input_variables)

    st.markdown("<h3 style='text-align: center; color: black;'>Conclusão do resultado apontado pelo modelo:</h3>", unsafe_allow_html=True)
    
    if yhat==0:
        st.success('De acordo com o modelo o paciente possui baixa probabiliade de ir para UTI!')
        
    else:
        st.error('De acordo como modelo o paciente possui alta probabilidade de ir para UTI!')
        
    st.markdown("<h3 style='text-align: center; color: black;'>Estatísticas do resultado:</h3>", unsafe_allow_html=True)
        
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Acurácia Total do Modelo Utilizado", "64,7 %")
    col2.metric("Resultado a Predição", int(yhat))
    if yhat_prob[0][0] > yhat_prob[0][1]:
        porc1 = '%'
        porc2 = '-%'
    else:
        porc1 = '-%'
        porc2 = '%'
        
    col3.metric("Probabilidade de ser negativo:", np.around(yhat_prob[0][0],2), porc1)
    col4.metric("Probabilidade de ser positivo:", np.around(yhat_prob[0][1],2), porc2)
    
    st.markdown("<h5 style='text-align: center; color: black;'>Importancia das variáveis de entrada para o modelo em operação</h5>", unsafe_allow_html=True)        
        
    image = Image.open('importancias_var_random_forest.png')
    st.image(image)
    
    st.markdown("<h5 style='text-align: center; color: black;'>Matriz de confusão para o modelo em operação</h5>", unsafe_allow_html=True)
    image = Image.open('matriz_confusao_random_forest.png')
    st.image(image)
    
    

        

        
        
        
        
                
        
        

        
         
    


    