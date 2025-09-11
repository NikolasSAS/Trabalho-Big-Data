import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dados
X = np.array([19, 20, 21, 22, 23, 24]).reshape(-1, 1)  
y = np.array([6.65, 6.58, 7.01, 5.34, 6.32, 6.51])     

# Modelo de regressão linear
modelo = LinearRegression()
modelo.fit(X, y)

# Previsões
y_pred = modelo.predict(X)

# Exibir os coeficientes
print(f'Coeficiente angular (inclinação): {modelo.coef_[0]:.2f}')
print(f'Coeficiente linear (intercepto): {modelo.intercept_:.2f}')

# Criar uma figura com 2 subgráficos
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Subgráfico 1: Regressão Linear
axs[0].scatter(X, y, color='blue', label='Média de computadores')
axs[0].plot(X, y_pred, color='red', label='Linha de regressão')
axs[0].set_title('Regressão Linear Simples')
axs[0].set_xlabel('X')
axs[0].set_ylabel('y')
axs[0].legend()
axs[0].grid(True)

# Subgráfico 2: Distribuição de Frequência (Histograma)
axs[1].hist(y, bins=5, color='darkblue', edgecolor='black')
axs[1].set_title('Distribuição de Frequência')
axs[1].set_xlabel('Média de computadores')
axs[1].set_ylabel('Frequência')
axs[1].grid(True)

plt.tight_layout()
plt.show()