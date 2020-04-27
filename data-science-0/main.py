#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


black_friday.head()


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[4]:


def q1():
    # Retorne aqui o resultado da questão 1.
    #Função shape retorna tupla com o n_observações e n_colunas
    return(black_friday.shape)


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[5]:


def q2():
    # Retorne aqui o resultado da questão 2.
    # Fitlro das linhas atráves do loc das duas condições (Age e Gender) 
    return int(black_friday.loc[(black_friday.Age == '26-35') & (black_friday.Gender == 'F'),'User_ID'].size)


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[15]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return int(black_friday.User_ID.unique().size)


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[16]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return black_friday.dtypes.unique().size


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[17]:


def q5():
    # Retorne aqui o resultado da questão 5.
    # Quantidade de linhas com pelo menos uma variável nula
    linhas_var_null  = black_friday[black_friday.isnull().any(axis=1)].shape[0]
    # Total de linhas do dataset
    total_linhas = black_friday.shape[0]
    return linhas_var_null/total_linhas 


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[18]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return black_friday.isnull().sum().max()


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[19]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return black_friday.Product_Category_3.mode().item()


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[20]:


def q8():
    # Retorne aqui o resultado da questão 8.
    # Realiza a normalização dos dados (Xi - Xmin / Xmax - Xmin)
    black_friday_purchase_norm = (black_friday.Purchase - black_friday.Purchase.min()) / (black_friday.Purchase.max() - black_friday.Purchase.min())
    # Retorna a média dos dados normalizados
    return float(black_friday_purchase_norm.mean())


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[21]:


def q9():
    # Retorne aqui o resultado da questão 9.
    # Realiza a padronização dos dados (Xi - Média / Desvio Padrão)
    black_friday_Purchase_padronizacao = (black_friday.Purchase - black_friday.Purchase.mean()) / black_friday.Purchase.std()
    # Retorna soma de valores entre -1 e 1 inclusive
    return int(black_friday_Purchase_padronizacao.between(-1,1).sum())
  


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[22]:


def q10():
    # Retorne aqui o resultado da questão 10.
    # Testa se para mesma linha as duas colunas(`Product_Category_2` e 'Product_Category_3') e gera uma lista com retorno da comparação
    testa = black_friday['Product_Category_2'].isnull() == black_friday['Product_Category_3'].isnull() 
    # Verifica se para alguma linha o retorno foi True
    return(True in testa)
    

