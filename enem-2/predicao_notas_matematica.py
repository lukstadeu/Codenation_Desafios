import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error


test = pd.read_csv("test.csv", sep=',', index_col=False)
train = pd.read_csv("train.csv",sep=',', index_col=False)

# Removendo coluna desnecessária do dataset de treino
train = train.drop(['Unnamed: 0'], axis=1)


# Visualizando quantidade de entradadas e variáveis nos dataset de treino e teste
print(f'Dataset Treino: \nEntradas: \t {train.shape[0]} \nVariáveis: \t {train.shape[1]} \n')
print(f'Dataset Teste: \nEntradas: \t {test.shape[0]} \nVariáveis: \t {test.shape[1]}')

# Armazenando a coluna com o número da inscrição que será utilizado no arquivo de resposta
inscricao = test['NU_INSCRICAO']

"""
 Verificando se as variáveis presentes no dataset de treino estão presentes no dataset de teste e isso é Falso, pois o dataset de Treno tem 166 variáveis e enquanto 
o de teste tem somente 47.

"""
print(set(train.columns).issubset(set(test.columns)))

""" 
Neste caso, como o dataset de teste apresenta somente 47 variáveis entendemos que as variáveis preditoras estão dentre essas 47 variáveis, então podemos simplificar 
a análise utilizando somente com as variáveis presente no dataset de teste.

"""

# Obtendo colunas do dataset de teste
variaveis_test = list(test.columns)

# Adicionando a variável predita 
variaveis_test.append('NU_NOTA_MT')

train = train[variaveis_test]

# Verificando a correlação das variáveis com a nota de matemática
pd.DataFrame(train[train.columns[1:]].corr()['NU_NOTA_MT'][:-1]).sort_values(by = 'NU_NOTA_MT', ascending = False)

"""
As variáveis com maior relação são:
'NU_NOTA_CN' - Nota da prova de Ciências da Natureza
'NU_NOTA_CH' - Nota da prova de Ciências Humanas
'NU_NOTA_LC' - Nota da prova de Linguagens e Códigos

Para o treinamento do nosso modelo iremos utilizar essas 3 variáveis.
"""

# Definindo as features
features = [
    'NU_NOTA_CN',
    'NU_NOTA_CH',
    'NU_NOTA_LC',
]


# Verificando os missing values para features no dataset de treino
train[features].isnull().sum()

# Preenchendo com o valor -1 as variáveis preditoras que estão com valores faltantes no dataset de treino 
train['NU_NOTA_CN'] = train['NU_NOTA_CN'].fillna(-1)
train['NU_NOTA_CH'] = train['NU_NOTA_CH'].fillna(-1)
train['NU_NOTA_LC'] = train['NU_NOTA_LC'].fillna(-1)

# Preenchendo com o valor 0 a nota de matemática que não está preenchida no dataset de treino
train['NU_NOTA_MT'] = train['NU_NOTA_MT'].fillna(0)

#train['NU_NOTA_MT'].isnull().sum()

# Verificando os missing values para features no dataset de teste
test[features].isnull().sum()

# Preenchendo com o valor -1 as variáveis preditoras que estão com valores faltantes no dataset de teste
test['NU_NOTA_CN'] = test['NU_NOTA_CN'].fillna(-1)
test['NU_NOTA_CH'] = test['NU_NOTA_CH'].fillna(-1)
test['NU_NOTA_LC'] = test['NU_NOTA_LC'].fillna(-1)

# Armazenando a nota de matemática (variável predita)
y = train['NU_NOTA_MT']

# Armazenando as features (variáveis preditoras) nos dataset de treino e teste
x_train = train[features]
x_test = test[features]

x_train.shape

# Padronizando as features
#scaler = StandardScaler().fit(x_train)

X_train_scaled = StandardScaler().fit_transform(x_train)
X_test_scaled = StandardScaler().fit_transform(x_test)


# Instanciando o modelo de RandomForest com 100 estimadores e profundidade máxima da árvore =7
rfr = RandomForestRegressor(n_estimators=100,max_depth=7)

# Realizando o ajuste do modelo com os dados de treino
rfr.fit(X_train_scaled , y)

# Realizado a predição das notas
pred_notas = rfr.predict(X_test_scaled)


# Gerando o dataset de resposta com o nu_inscrição e a nota predita
resultado = pd.DataFrame(list(zip(inscricao,pred_notas)),columns=['NU_INSCRICAO','NU_NOTA_MT'])

# Salvando a resposta em um arquivo .csv
resultado.to_csv('answer.csv',encoding='UTF-8',index=False)