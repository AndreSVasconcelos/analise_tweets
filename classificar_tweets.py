# Libs
from classes.bases_treinamento import BaseTreinamento
from classes.modelos_treinamento import ModeloTreinamento
from classes.bases_teste import BaseTeste
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

# Criar objeto da base de treinamento
print('Carregando base de treinamento...')
base_de_treinamento = BaseTreinamento(pd.read_csv('./tweets/Train50_short.csv', 
                                                  encoding='utf-8', 
                                                  on_bad_lines='skip', 
                                                  sep=';'))

# Criar modelo de treinamento
print('Carregando modelo de treinamento...')
modelo_de_treinamento = ModeloTreinamento(base_de_treinamento.bd_treinamento)

# Avaliar historico do treinamento
print('Avaliando historico do treinamento...')
historico_loss = []
for i in modelo_de_treinamento.historico:
    historico_loss.append(i.get('textcat'))
historico_loss = np.array(historico_loss)
plt.plot(historico_loss)
plt.title('Progressão do erro')
plt.xlabel('Épocas')
plt.ylabel('Erro')

# Testar modelo
print('Carregando base de teste...')
base_de_teste = BaseTeste(pd.read_csv('./tweets/Test_short.csv', 
                                      encoding='utf-8',
                                      on_bad_lines='skip',
                                      sep=';'))
print('Fazendo previsões...')
previsoes = []
for tweet_text in base_de_teste.bd_teste['tweet_text']:
    previsao = modelo_de_treinamento.modelo(tweet_text)
    previsoes.append(previsao.cats)

previsoes_final = []
for i in previsoes:
    if i['POSITIVO'] > i['NEGATIVO']:
        previsoes_final.append(1)
    else:
        previsoes_final.append(0)

previsoes_final = np.array(previsoes_final)
respostas_reais = base_de_teste.bd_teste['sentiment'].values
accuracy_score(respostas_reais, previsoes_final)
cm = confusion_matrix(respostas_reais, previsoes_final)
print('============================================================')
print('Confusion matrix da base de teste:')
print(cm)