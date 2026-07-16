# Previsão de Preços das Ações do Google com LSTM

Este projeto utiliza uma **Rede Neural Recorrente do tipo LSTM (Long Short-Term Memory)** para prever o preço de abertura das ações do Google com base em dados históricos.

O modelo aprende padrões temporais presentes na série histórica das ações e utiliza os **60 dias anteriores** para prever o preço de abertura do dia seguinte.

O projeto demonstra todo o fluxo de um problema de previsão de séries temporais utilizando **Deep Learning**, incluindo pré-processamento dos dados, normalização, treinamento da rede neural, geração de previsões e visualização gráfica dos resultados.

---

# Funcionalidades

- Leitura da base histórica de ações do Google.
- Normalização dos dados utilizando `MinMaxScaler`.
- Criação de janelas deslizantes para treinamento da LSTM.
- Construção de uma Rede Neural Recorrente com múltiplas camadas LSTM.
- Aplicação de Dropout para reduzir overfitting.
- Predição dos preços de abertura das ações.
- Comparação gráfica entre os valores reais e previstos.

---

# Estrutura do Projeto

```text
google-stock-price-prediction/
│
├── main.py
├── Google_Stock_Price_Train.csv
├── Google_Stock_Price_Test.csv
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Tecnologias Utilizadas

- Python 3
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- TensorFlow
- Keras

---

# Bibliotecas Utilizadas

- **Pandas**
  - Manipulação e leitura dos conjuntos de dados.

- **NumPy**
  - Operações matemáticas e manipulação de arrays.

- **Matplotlib**
  - Construção dos gráficos de comparação entre valores reais e previstos.

- **Scikit-Learn**
  - Normalização dos dados utilizando `MinMaxScaler`.

- **TensorFlow / Keras**
  - Construção, treinamento e execução da Rede Neural LSTM.

---

# Base de Dados

O projeto utiliza dois arquivos:

```text
Google_Stock_Price_Train.csv
Google_Stock_Price_Test.csv
```

A coluna utilizada para treinamento é:

```text
Open
```

que representa o preço de abertura diário da ação.

---

# Pré-processamento dos Dados

Antes do treinamento, são realizadas algumas etapas importantes.

## Seleção da variável

O modelo utiliza apenas a coluna **Open**.

```python
training_set = dataset_train.iloc[:,1:2].values
```

---

## Normalização

Os valores são normalizados para o intervalo entre **0 e 1** utilizando:

```python
MinMaxScaler(feature_range=(0,1))
```

A normalização melhora o desempenho da Rede Neural durante o treinamento.

---

## Janela Deslizante

A LSTM recebe como entrada os **60 dias anteriores** para prever o próximo valor.

Exemplo:

```text
Dias 1 até 60  → prevê o dia 61

Dias 2 até 61  → prevê o dia 62

Dias 3 até 62  → prevê o dia 63
```

Esse método permite que a rede aprenda relações temporais entre os preços das ações.

---

# Arquitetura da Rede Neural

A rede foi construída utilizando a API Sequencial do Keras.

Ela possui:

- Camada de Entrada
- 4 camadas LSTM
- 4 camadas Dropout
- 1 camada Dense de saída

Configuração:

```python
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
```

---

# Treinamento do Modelo

Após a criação da arquitetura, o modelo deve ser compilado e treinado.

Exemplo:

```python
modelo.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

modelo.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32
)
```

Também pode ser utilizado o callback `EarlyStopping` para interromper o treinamento quando não houver melhora na função de perda.

---

# Processo de Previsão

Para realizar as previsões:

1. Os dados de treino e teste são unidos.
2. São selecionados os últimos 60 dias anteriores ao período de teste.
3. Os dados são normalizados utilizando o mesmo `MinMaxScaler`.
4. São criadas novas janelas deslizantes.
5. A Rede Neural gera as previsões.
6. Os valores previstos são convertidos novamente para a escala original.

```python
predicted_stock_price = modelo.predict(X_test)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)
```

---

# Visualização dos Resultados

Ao final da execução é exibido um gráfico comparando:

- Linha vermelha: preços reais das ações.
- Linha azul: preços previstos pelo modelo.

Exemplo:

```python
plt.plot(real_stock_price, color='red', label='Dados Reais de Ações do Google')
plt.plot(predicted_stock_price, color='blue', label='Dados Previstos de Ações do Google')

plt.title('Previsões de Ações do Google')
plt.xlabel('Tempo')
plt.ylabel('Preço das Ações')

plt.legend()
plt.show()
```

---

# Como Configurar o Projeto

## 1. Clonar o repositório

```bash
git clone https://github.com/SEU_USUARIO/google-stock-price-prediction.git

cd google-stock-price-prediction
```

---

## 2. Criar um ambiente virtual (Opcional)

### Linux/macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

---

## 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

---

# Como Executar

Execute:

```bash
python main.py
```

O programa irá:

1. Carregar os dados de treinamento.
2. Normalizar os valores.
3. Criar as sequências temporais.
4. Construir a Rede Neural LSTM.
5. Treinar o modelo.
6. Carregar os dados de teste.
7. Realizar as previsões.
8. Exibir o gráfico comparando os preços reais e previstos.

---

# Fluxo do Projeto

```text
Leitura da Base de Treinamento
            │
            ▼
Seleção da coluna Open
            │
            ▼
Normalização dos Dados
            │
            ▼
Criação das Janelas Deslizantes
            │
            ▼
Construção da Rede LSTM
            │
            ▼
Treinamento do Modelo
            │
            ▼
Leitura da Base de Teste
            │
            ▼
Preparação dos Dados
            │
            ▼
Previsão dos Preços
            │
            ▼
Desnormalização
            │
            ▼
Comparação entre Valores Reais e Previstos
            │
            ▼
Exibição do Gráfico
```

---

# Melhorias Futuras

- Salvar o modelo treinado para evitar novos treinamentos.
- Ajustar automaticamente hiperparâmetros da rede.
- Utilizar mais variáveis, como High, Low, Close e Volume.
- Implementar métricas como RMSE e MAE.
- Realizar previsões para múltiplos dias futuros.

---

# Autor

Projeto desenvolvido para fins de estudo e prática de Deep Learning aplicado à previsão de séries temporais utilizando Redes Neurais LSTM, TensorFlow/Keras e Python.