#!/usr/bin/env python
# coding: utf-8

# # Introdução 

# Nos dias atuais,tecnologia e finanças vem andando cada vez mais juntos,com inúmeras aplicações da tecnologia em instituições financeiras . Uma das aplicações que se destaca ,em especial em bancos e instituições financeiras , é  o credit score,que tem como objetivo determinar se uma pessoas possui ou não perfil para adquirir crédito . Para construir o credit score,é utilizado desde técnicas de Machine Learning à Análise Estatística . Neste projeto,será construido modelos de Machine Learning para verificar se uma pessoa possui um "bom" ou "mau" risco de crédito,sendo que o conjunto de dados utilizado será "german_credit_data.csv",que contém informações de empréstimo de um banco alemão . Pode-se então conhecer de uma maneira mais completa esse conjunto de dados . 
# 

# # Dicionário dos dados 

# **Age**=Idade do solicitante do empréstimo 
# 
# **Sex** = Sexo do solicitante do empréstimo,(Variável Categórica - Nos níveis "Male"(Masculino) e "female"(Feminino)
# 
# **Job** = Tipo de emprego que o solicitante possui,(Variável Categórica - Nos níveis  "0 - unskilled(Não qualificado) e non-resident(Não residente)","1 - unskilled(Qualificado) and resident(Residente)"," 2 - skilled(Especializada)"," 3 - highly skilled(Altamente especializada)")
# 
# **Housing** = Tipo de casa em que o solicitante mora ,(Variável Categórica - nos níveis own (Própria), rent(Aluguel), or free(Gratuita))
# 
# **Saving accounts** = Contas salvas(Variável Cetegórica - little(Pequena), moderate(Moderada), quite rich(Muito rico), rich(Rico))
# 
# **Checking account** = Conta corrente do solicitante ,Variável quantitativa discreta
# 
# **Credit amount** = Quantidade de crédito 
# 
# **Duration**= Duração do empréstimo em meses
# 
# **Purpose** = Propóposito do empréstimo (Variável Categórica - car(Carro),furniture/equipment(Móveis/Equipamentos), r
# adio/TV, domestic appliances(Eletrodomésticos), repairs(Reparos), education(Educação), business(Negócios), vacation/others(Férias))

# Pode-se observar as colunas que o banco de dados contém a partir das 5 primeiras linhas,e também o número de colunas e linhas que o banco de dados possui : 

# In[94]:


#Importação das bibliotecas a serem utilizadas 
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
dados=pd.read_csv("german_credit_data.csv")
dados.head()


# In[95]:


dados.shape


# Pode-se observar que o banco de dados é composto por 1000 linhas,e 11 colunas . Porém a coluna "Unnamed: 0" não é significante para a análise, pois ela contém o id da observação, logo não traz nenhuma informação relevante, logo pode-se retira-la dos dados . 

# # Pré processamento dos dados e limpeza dos dados 

# In[96]:


dados.drop(columns=["Unnamed: 0"],inplace=True)


# Agora pode-se observar como o Pandas em que tipo cada variável do conjunto de dados,e verificar se ela está adequada : 

# In[97]:


dados.dtypes 


# Outro fator relevante antes de fazer uma análise estatistica é verificar se o conjunto de dados contém valores ausentes : 

# In[98]:


dados.isnull().sum()/len(dados)*100


# É possivel observar que a variável "Saving accounts" é composta de 18.3% das observações de  valores ausentes ,enquanto que a variável "Checking Account" possui 39.4% das observações ausentes . Logo,pode-se observar quais os valores mais frequentes nessas colunas,e então tomar uma decisão para substituir os valores ausentes , tal como, substituir pelo valor mais frequente, remover linhas que contém Na, etc, porém a decisão tomada tem que ser analisada caso a caso . 

# In[99]:


dados["Saving accounts"].value_counts()


# In[100]:


dados["Checking account"].value_counts()


# Para a coluna "Checking account" foi obtido que existe um equilibrio entre a resposta "litte" e "moderate",logo não há um valor que diferencia dos demais , logo uma estratégia que pode ser utilizada para os valores ausentes é  tratar as observações ausentes da coluna como um nível de resposta,e substitui-los pelo valor "Não Informado" . Enquanto que na variável "Saving accounts" cerca de 60% das observações são little , logo uma possível estratégia é substituir os valores ausentes dessa coluna por "little" .

# In[101]:


dados["Saving accounts"].fillna("little",inplace=True)
dados["Checking account"].fillna("Não informado",inplace=True)


# # Análise exploratória

# Agora que os valores ausentes já foram tratados é interessante conhecer quantas observações possuem risco alto e risco baixo a partir de um gráfico de barras : 

# In[102]:


sns.countplot(dados["Risk"])


# Ao se observar o gráfico, pode-se observar que o número de observações que possui um risco considerado bom é de 70%  das observações, o que indica que a maior parte dos pagadores tem um bom nível de chance de pagar o crédito . Pode-se então conhecer as medidas sumárias e as distribuição dos dados das variáveis quantitativas do Dataset  : 

# In[103]:


dados[["Age","Credit amount","Duration"]].describe()


# In[104]:


fig,(axis1,axis2,axis3)=plt.subplots(1,3,figsize=(18,6))
sns.boxplot(dados["Age"],ax=axis1)
sns.boxplot(dados["Credit amount"],ax=axis2)
sns.boxplot(dados["Duration"],ax=axis3)


# In[88]:


dados[["Age","Credit amount","Duration"]].hist(figsize=(12,8));


# Das distribuições pode-se observar que nenhuma variável quantitativa aparenta possuir uma distribuição normal . Enquanto que da tabela pode-se observar que aparentemente não há nenhum valor discrepante,que foge a normalidade . 

# In[105]:


fig,(axis1,axis2,axis3)=plt.subplots(1,3,figsize=(20,8))
axis1.bar(dados.groupby("Risk")["Age"].mean().index,dados.groupby("Risk")["Job"].mean().values)
axis1.set_title("Média de idade de acordo com o tipo de risco")
axis2.bar(dados.groupby("Risk")["Duration"].mean().index,dados.groupby("Risk")["Duration"].mean().values)
axis2.set_title("Média de duração do empréstimo de acordo com o tipo de risco")
axis3.bar(dados.groupby("Risk")["Credit amount"].mean().index,dados.groupby("Risk")["Credit amount"].mean().values)
axis3.set_title("Média de crédito amount de acordo com o tipo de risco")


# Aparentemente não há uma relação considerável entre o Risco associado e as variáveis Age,e Job,pois para os dis fatores de risco dessa variável o valor médio aparenta ser o mesmo . Já para as variáveis Duration e Credit Amount o valor médio dessas variáveis aparenta ser diferente para os dois níveis de risco . 

# In[106]:


from sklearn.preprocessing import LabelEncoder


# In[107]:


labelencoder=LabelEncoder()
dados.iloc[:,9]=labelencoder.fit_transform(dados.iloc[:,9])


# In[108]:


fig,(axis1,axis2)=plt.subplots(1,2,figsize=(12,6))
sns.barplot(dados["Sex"],y=dados["Risk"],ax=axis1)
sns.barplot(x=dados["Housing"],y=dados["Risk"],ax=axis2)
fig,(axis1,axis2)=plt.subplots(1,2,figsize=(12,6))
sns.barplot(x=dados["Saving accounts"],y=dados["Risk"],ax=axis1)
sns.barplot(dados["Checking account"],y=dados["Risk"],ax=axis2)


# Ao se observar o gráfico de risco dado por sexo,é possivel observar que pessoas do sexo masculino aparenta possuir um histórico de crédito levemente melhor do que pessoas do sexo feminino,embora a diferença não seja tão grande . Ao se analisar o risco dado pelo tipo de casa,pode-se observar que pessoas que possuiem casa própria apresenta um risco menor,enquanto que moram de aluguel apresentam um risco maior .  Pode-se fazer uma análise semelhante para a variável "Purpose" ,, 

# In[109]:


plt.figure(figsize=(12,8))
sns.barplot(x=dados["Purpose"],y=dados["Risk"])


# Ao se observar o gráfico de risco dado por sexo,é possivel observar que pessoas do sexo masculino aparenta possuir um histórico de crédito levemente melhor do que pessoas do sexo feminino,embora a diferença não seja tão grande . Ao se analisar o risco dado pelo tipo de casa,pode-se observar que pessoas que possuiem casa própria apresenta um risco menor,enquanto que moram de aluguel apresentam um risco maior .  

# # Construção dos modelos de machine learning 
# 

# Por fim,pode-se construir os modelos Machine Learning para classificar o risco de crédito . 

# In[276]:


#Construção de listas para armazenar as acurácias dos modelos e a lista dos algortimos 
listaalgoritmos=[]
listaacuracia=[]


# In[277]:


x=dados.iloc[:,0:9].values
y=dados.iloc[:,9].values 


# In[278]:


labelencoder=LabelEncoder()
#Sexo = 1 se masculino,0 se feminino
#Housing=1 se própria ,0 se free ,2 se aluguel
#Saving account =0 se little,1 se moderate,2 se quite rich ,3 se rich 
#Checking account = 1 se little,2 se moderate,0 se não informado,3 se rich 
#Purpose= 5 se Radio/Tv,3 se Education,4 se furniture/equipment,1 se car,0 se business 
x[:,1]=labelencoder.fit_transform(x[:,1])
x[:,3]=labelencoder.fit_transform(x[:,3])
x[:,4]=labelencoder.fit_transform(x[:,4])
x[:,5]=labelencoder.fit_transform(x[:,5])
x[:,8]=labelencoder.fit_transform(x[:,7])


# In[279]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score


# In[280]:


column_transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [1,2,3,4,5,8])],remainder='passthrough')


# In[281]:


x=column_transformer.fit_transform(x).toarray()


# In[282]:


modelonaive=GaussianNB()
acuracia=cross_val_score(modelonaive,x,y,cv=5,scoring="accuracy")
listaacuracia.append(acuracia.mean())
listaalgoritmos.append("Naive Bayes")


# In[283]:


from sklearn.linear_model import LogisticRegression


# In[284]:


modeloregressao=LogisticRegression(solver="lbfgs",max_iter=1000)


# In[285]:


acuracia=cross_val_score(modeloregressao,x,y,cv=5,scoring="accuracy")
listaacuracia.append(acuracia.mean())
listaalgoritmos.append("Regressão Logistica")


# In[286]:


from sklearn.ensemble import RandomForestClassifier 


# In[287]:


modelorandomforest=RandomForestClassifier(n_estimators=40,criterion="entropy")


# In[288]:


acuracia=cross_val_score(modelorandomforest,x,y,cv=5,scoring="accuracy")
listaacuracia.append(acuracia.mean())
listaalgoritmos.append("Random Forest")


# In[289]:


from sklearn.tree import DecisionTreeClassifier


# In[290]:


modeloarvore=DecisionTreeClassifier(criterion="entropy")


# In[291]:


acuracia=cross_val_score(modeloarvore,x,y,cv=5,scoring="accuracy")
listaacuracia.append(acuracia.mean())
listaalgoritmos.append("Árvore de Decisão")


# In[292]:


from sklearn.svm import SVC


# In[293]:


modelosvm=SVC(kernel="linear")
listaacuracia.append(acuracia.mean())
listaalgoritmos.append("SVM")


# In[294]:


from sklearn.preprocessing import StandardScaler


# In[295]:


standardscaler=StandardScaler()
x=standardscaler.fit_transform(x)


# In[296]:


from sklearn.neighbors import KNeighborsClassifier 


# In[297]:


modeloknn=KNeighborsClassifier(n_neighbors=60)
acuracia=cross_val_score(modeloknn,x,y,cv=5,scoring="accuracy")
listaacuracia.append(acuracia.mean())
listaalgoritmos.append("KNN")


# In[298]:


dadosresultados=pd.DataFrame()
dadosresultados["Algoritmo"]=listaalgoritmos
dadosresultados["Acurácia"]=listaacuracia


# Então,pode-se conhecer a acurácia média dos algoritmos por meio de uma tabela .

# In[301]:


dadosresultados=dadosresultados.sort_values(by="Acurácia",ascending=False)
dadosresultados


# Ao se observar a tabela pode-se verificar que o algoritmo que teve uma melhor acurácia média para classificar se um pessoa tem um bom ou alto risco de crédito foi Regressão Logistica e Rndom Forest,sendo a diferenças do resultado desses dois algoritmos quase insignificante . Além disso,o algoritmo que teve a menor acurácia média foi Naive Bayes . Pode-se visualizar também os resultados de forma gráfica . 

# In[302]:


plt.figure(figsize=(12,8))
plt.bar(dadosresultados["Algoritmo"],dadosresultados["Acurácia"])
plt.xlabel("Algoritmo")
plt.ylabel("Acurácia Média")
plt.title("Acurácia média de acordo com o algoritmo")


# In[ ]:




