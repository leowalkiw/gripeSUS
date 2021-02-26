# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 18:51:27 2021

@author: T-Gamer
"""

import pandas as pd
import numpy as np

#carregar a base
base = pd.read_csv('dados-pr-2.csv', error_bad_lines=False, sep=';', encoding='latin-1')


base.drop('id', 1, inplace=True)
base.drop('dataInicioSintomas', 1, inplace=True)
base.drop('dataNotificacao', 1, inplace=True)
base.drop('dataNascimento', 1, inplace=True)
base.drop('cbo', 1, inplace=True)
base.drop('condicoes', 1, inplace=True)
base.drop('dataTeste', 1, inplace=True)
base.drop('tipoTeste', 1, inplace=True)
base.drop('paisOrigem', 1, inplace=True)
base.drop('estadoIBGE', 1, inplace=True)
base.drop('municipioIBGE', 1, inplace=True)
base.drop('excluido', 1, inplace=True)
base.drop('origem', 1, inplace=True)
base.drop('cnes', 1, inplace=True)
base.drop('estadoNotificacao', 1, inplace=True)
base.drop('estadoNotificacaoIBGE', 1, inplace=True)
base.drop('municipioNotificacao', 1, inplace=True)
base.drop('municipioNotificacaoIBGE', 1, inplace=True)
base.drop('validado', 1, inplace=True)
base.drop('dataEncerramento', 1, inplace=True)
base.drop('classificacaoFinal', 1, inplace=True)


#apagando os registros da coluna evolucao que estão como cancelado
base.drop(base[base.evolucaoCaso == 'Cancelado'].index, inplace=True)

#apagando registros undefined
base.drop(base[base.profissionalSaude == 'undefined'].index, inplace=True)
base.drop(base[base.idade == 'undefined'].index, inplace=True)


#convertendo coluna em numerico
base.idade = pd.to_numeric(base.idade , errors='coerce')
#apagando registros com idade maior que 118, pessoa mais velha do mundo
base.drop(base[base.idade >= 118].index, inplace=True)



#alterando coluna sintomas, tratar o valor Outros: Paciente assintomático
base.loc[base.sintomas == 'Outros: Paciente assintomático', 'sintomas'] = 'Assintomático'

#não sera usado mais, sera inserido o valor mais frequente
#apagando os registros da coluna evolucao que estão como nan
#base.drop(base[pd.isnull(base['evolucaoCaso'])].index, inplace=True)


#substituindo pelo valor mais frequente da coluna
base.sintomas.fillna(base.sintomas.mode()[0], inplace=True)

#substituindo pelo valor mais frequente da coluna
base.profissionalSaude.fillna(base.profissionalSaude.mode([0]), inplace=True)

#substituindo pelo valor mais frequente da coluna
base.estadoTeste.fillna(base.estadoTeste.mode()[0], inplace=True)

#substituindo pelo valor mais frequente da coluna
base.resultadoTeste.fillna(base.resultadoTeste.mode()[0], inplace=True)

#substituindo pelo valor mais frequente da coluna
base.sexo.fillna(base.sexo.mode()[0], inplace=True)

#substituindo pelo valor mais frequente da coluna
base.estado.fillna(base.estado.mode()[0], inplace=True)

#substituindo pelo valor mais frequente da coluna
base.municipio.fillna(base.municipio.mode()[0], inplace=True)

#substituindo pelo valor mais frequente da coluna
base.idade.fillna(base.idade.mode()[0], inplace=True)

#substituindo pelo valor mais frequente da coluna
base.evolucaoCaso.fillna(base.evolucaoCaso.mode()[0], inplace=True)



#separando os atributos previsores
previsores = base.iloc[:, 0:8].values


#criando a classe
classe = base.iloc[:, 8]


#para trabalhar com o algoritmo, precisa transformar os atributos em numeros
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()


#transformando cada coluna dos previsores em numeros
previsores[:, 0] = labelencoder.fit_transform(previsores[:,0])
previsores[:, 1] = labelencoder.fit_transform(previsores[:,1])
previsores[:, 2] = labelencoder.fit_transform(previsores[:,2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:,3])
previsores[:, 4] = labelencoder.fit_transform(previsores[:,4])
previsores[:, 5] = labelencoder.fit_transform(previsores[:,5])
previsores[:, 6] = labelencoder.fit_transform(previsores[:,6])


#importando o Naive Bayes
from sklearn.naive_bayes import GaussianNB

classificador = GaussianNB()

classificador.fit(previsores, classe)
#treinamento do algoritmo, gerando a tabela de probabilidade com o metodo fit, 

#criando um exemplo a ser classificado
resultado = classificador.predict([[299,0,0,2,2,25,55,40]])


print(resultado)

