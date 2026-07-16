import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Input
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping


dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
dataset_train

# separa a coluna open, converte para numpy
training_set = dataset_train.iloc[:,1:2].values
training_set

# normalização entre zero e 1
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)
training_set_scaled


# janela deslizante - Necessário para LSTM
# necessário: amostras, timesteps, features
# X contém amostras, cada amostra 60 valores, 1198 amostras (1258 -60)
# y contém o dia seguinte
X_train = []
y_train = []
for i in range(60, 1258):
  X_train.append(training_set_scaled[i-60:i,0])
  y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

print(X_train)

# finalmente amostras, timestesps, features
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

modelo = Sequential([
    Input(shape=(X_train.shape[1], 1)),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)
])

dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
#coluna open
real_stock_price = dataset_test.iloc[:,1:2].values 


#junta treino e teste na mesma série
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
# pega 60 dias anteriores ao periodo de previsão
inputs = dataset_total[len(dataset_total)-len(dataset_test) - 60:].values
inputs

# transforma em formato 2D
inputs = inputs.reshape(-1,1)
inputs


# aplicamos a mesma normalização
inputs = sc.transform(inputs)
# mesma lógica do treino - 20 amostras
X_test = []
for i in range(60,80):
  X_test.append(inputs[i-60:i,0])
# converte para numpy
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
# faz a previsões
predicted_stock_price = modelo.predict(X_test)
# desnormaliza
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color='red', label="Dados Reais de Ações do Google")
plt.plot(predicted_stock_price, color='blue', label="Dados Previstos de Ações do Google")
plt.title("Previsões de Ações do Google")
plt.xlabel("Tempo")
plt.ylabel("Preços de Ações do Google")
plt.legend()
plt.show()