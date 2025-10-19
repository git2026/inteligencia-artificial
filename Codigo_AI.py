# Executámos o código no Google Colab por isso é que está aqui um pip install. se quizer pode remove-lo
!pip install datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from datasets import load_dataset

# Carregar o dataset e obter dados necessários
conjunto_dados = load_dataset("ErfanMoosaviMonazzah/fake-news-detection-dataset-English")
treino_df = pd.DataFrame(conjunto_dados['train'])
teste_df = pd.DataFrame(conjunto_dados['test'])

# Remover linhas com valores nulos e separar as labels dos textos
treino_df.dropna(inplace=True)
teste_df.dropna(inplace=True)
texto_treino = treino_df['text']
etiqueta_treino = treino_df['label']
texto_teste = teste_df['text']
etiqueta_teste = teste_df['label']

# Vetorização
vetorizador = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
texto_treino_vetorizado = vetorizador.fit_transform(texto_treino)
texto_teste_vetorizado = vetorizador.transform(texto_teste)

# Modelos
modelos = {
    'Regressão Logística': LogisticRegression(max_iter=1000),
    'SVM': SVC(kernel='linear', C=1),
    'Naive Bayes': MultinomialNB(),
    'Floresta Aleatória': RandomForestClassifier(n_estimators=100)
}
# Treino e avaliação com os modelos
for nome_modelo, modelo in modelos.items():
    print(f"Treinando {nome_modelo}...")
    modelo.fit(texto_treino_vetorizado, etiqueta_treino)
    predicao_etiqueta = modelo.predict(texto_teste_vetorizado)
    print(f"\nRelatório de Classificação para {nome_modelo}:")
    print(classification_report(etiqueta_teste, predicao_etiqueta, digits=4))
    matriz_confusao = confusion_matrix(etiqueta_teste, predicao_etiqueta)
    matriz = ConfusionMatrixDisplay(confusion_matrix=matriz_confusao)
    matriz.plot(cmap='Blues')
    plt.title(f"Matriz de Confusão - {nome_modelo}")
    plt.show()

resultados = {}
for nome_modelo, modelo in modelos.items():
    precisao = modelo.score(texto_teste_vetorizado, etiqueta_teste)
    resultados[nome_modelo] = precisao
plt.figure(figsize=(8, 5))
plt.bar(resultados.keys(), resultados.values(), color=['blue', 'green', 'red', 'orange'])
plt.title("Comparação de Precisão entre Modelos")
plt.ylabel("Precisão")
plt.show()