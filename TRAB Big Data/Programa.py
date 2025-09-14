import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df1 = pd.read_excel('Escolas Niteroi(2019).xlsx')
soma1 = df1['QT_DESKTOP_ALUNO'].sum() 
quantidade1 = df1['QT_DESKTOP_ALUNO'].count()
media1 = soma1 / quantidade1

df2 = pd.read_excel('Escolas Niteroi(2020).xlsx')
soma2 = df2['QT_DESKTOP_ALUNO'].sum() 
quantidade2 = df2['QT_DESKTOP_ALUNO'].count()
media2 = soma2 / quantidade2

df3 = pd.read_excel('Escolas Niteroi(2021).xlsx')
soma3 = df3['QT_DESKTOP_ALUNO'].sum() 
quantidade3 = df3['QT_DESKTOP_ALUNO'].count()
media3 = soma3 / quantidade3

df4 = pd.read_excel('Escolas Niteroi(2022).xlsx')
soma4 = df4['QT_DESKTOP_ALUNO'].sum() 
quantidade4 = df4['QT_DESKTOP_ALUNO'].count()
media4 = soma4 / quantidade4

df5 = pd.read_excel('Escolas Niteroi(2023).xlsx')
soma5 = df5['QT_DESKTOP_ALUNO'].sum() 
quantidade5 = df5['QT_DESKTOP_ALUNO'].count()
media5 = soma5 / quantidade5

df6 = pd.read_excel('Escolas Niteroi(2024).xlsx')
soma6 = df6['QT_DESKTOP_ALUNO'].sum() 
quantidade6 = df6['QT_DESKTOP_ALUNO'].count()
media6 = soma6 / quantidade6

# Dados
X = np.array([19, 20, 21, 22, 23, 24]).reshape(-1, 1)  
Y = np.array([media1, media2, media3, media4, media5, media6])
     

# Modelo de regressão linear
modelo = LinearRegression()
modelo.fit(X, Y)

# Previsões
y_pred = modelo.predict(X)

# Exibir os coeficientes
print(f'Coeficiente angular (inclinação): {modelo.coef_[0]:.2f}')
print(f'Coeficiente linear (intercepto): {modelo.intercept_:.2f}')

# Criar uma figura com 2 subgráficos
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Subgráfico 1: Regressão Linear
axs[0].scatter(X, Y, color='blue', label='Média de computadores')
axs[0].plot(X, y_pred, color='red', label='Linha de regressão')
axs[0].set_title('Regressão Linear Simples')
axs[0].set_xlabel('X')
axs[0].set_ylabel('y')
axs[0].legend()
axs[0].grid(True)

# Subgráfico 2: Distribuição de Frequência (Histograma)
axs[1].hist(Y, bins=5, color='darkblue', edgecolor='black')
axs[1].set_title('Distribuição de Frequência')
axs[1].set_xlabel('Média de computadores')
axs[1].set_ylabel('Frequência')
axs[1].grid(True)

plt.tight_layout()
plt.show()
