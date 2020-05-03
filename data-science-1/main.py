#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[3]:


#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[4]:


np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[5]:


norm = dataframe['normal']
binom = dataframe['binomial']


# In[6]:


# Sua análise da parte 1 começa aqui.

# Quartis distribuição normal
q1_norm = norm.quantile(0.25)
q2_norm = norm.quantile(0.50)
q3_norm = norm.quantile(0.75)

# Quartis distribuição binominal
q1_binom = binom.quantile(0.25) 
q2_binom = binom.quantile(0.50) 
q3_binom = binom.quantile(0.75) 


# Criando Dataframe com os quartis
quartis = pd.DataFrame({'normal' : [q1_norm,q2_norm,q3_norm],
                        'binomial' :[q1_binom,q2_binom,q3_binom]},
                        index = ['1_quartil','2_quartil','3_quartil'])

# Dataframe dos quartis
quartis


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[7]:


def q1():
    # Rertorne aqui o resultado da questão 1
    # Diferença entre os quartis das distribuições
    diferenca = [round(quartis.iloc[i,0] - quartis.iloc[i,1],3) for i in range(3)]

    return tuple(diferenca)


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?
# 

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[8]:


# Média e desvio padrão distribuição normal
media_norm = norm.mean()
desvio_padrao_norm = norm.std()

# Fit ECDF na distribuição normal
fit_norm  = ECDF(norm)

# Cácluclo da probabilidade
# Primeiro cálcula o valor ECDF para (média + desvio_padrão) e desse valor subtrai o valor calculado para o ECDF para (média - desvio_padrão)
probabilidade = round((fit_norm(media_norm + desvio_padrao_norm) - fit_norm(media_norm - desvio_padrao_norm)),3)


# In[136]:


def q2():
    # Rertorne aqui o resultado da questão 2
    return float(probabilidade)


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[9]:


# Média para distribuição normal já foi calculada para questão anteriror
# Variância da distribuição normal 
variancia_normal = norm.var()

# Variância e média da distribuição binomial
variancia_binomial = binom.var()
media_binomial = binom.mean()

# Diferença entre média e variância
resultado = (round(media_binomial - media_norm ,3),round(variancia_binomial - variancia_normal,3))


# In[10]:


def q3():
    # Rertorne aqui o resultado da questão 3
    return tuple(resultado)


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# ## Parte 2

# ### _Setup_ da parte 2

# In[11]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[17]:


# Sua análise da parte 2 começa aqui.

stars.head()


# In[72]:


stars['target'].value_counts()


# In[19]:


stars.describe()


# In[73]:


# Filtre apenas os valores de mean_profile onde target == 0 (ou seja, onde a estrela não é um pulsar).
false_pulsar = stars.loc[stars.target==0].mean_profile

# Padronize a variável mean_profile filtrada anteriormente para ter média 0 e variância 1.
false_pulsar_mean_profile_standardized = (false_pulsar - false_pulsar.mean()) / false_pulsar.std() 



# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[10]:


def q4():
    # Rertorne aqui o resultado da questão 4
    # Quantis teóricos obtidos através da função sct.norm.ppf
    quantis_teoricos = sct.norm.ppf([0.80,0.90,0.95], loc = 0 , scale = 1)

    ecdf =  ECDF(false_pulsar_mean_profile_standardized) 

    return tuple(ecdf(quantis_teoricos).round(3))


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[11]:


def q5():
    # Rertorne aqui o resultado da questão 5
    # Quartis distribuição normal de média 0 e variância 1
    quartis_norm = sct.norm.ppf([0.25,0.50,0.75], loc = 0, scale = 1)

    ecdf =  ECDF(false_pulsar_mean_profile_standardized) 

    # Quartis da variável 'false_pulsar_mean_profile_standardized'  
    quartis_ecdf = quartis_ecdf = np.quantile(false_pulsar_mean_profile_standardized,[0.25,0.50,0.75])

    return tuple((quartis_ecdf - quartis_norm).round(3))


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
