import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# Diretórios para as pastas com imagens de câncer e sem câncer
cancer_dir = r"C:\Users\Merop\Downloads\TCC 1\TCC\DatasetCancer"
sem_cancer_dir = r"C:\Users\Merop\Downloads\TCC 1\TCC\DatasetCancer"
diretorio_teste = r"C:\Users\Merop\Downloads\TCC 1\TCC\DatasetCancer\Teste"

# Função para carregar e processar imagens
def Carregar_Imagens(directory):
    images = []
    labels = []
    arq = []

    # Iteração pelas categorias 'Sem Cancer' e 'Com Cancer'
    for label, cancer_type in enumerate(['Sem Cancer', 'Com Cancer']):
        # Construção do caminho da pasta de imagens
        cancer_folder = os.path.join(directory, cancer_type)
        # Loop pelos arquivos na pasta de imagens
        for filename in os.listdir(cancer_folder):
            # Construção do caminho completo da imagem
            img_path = os.path.join(cancer_folder, filename)

            # Verificação se é um arquivo válido
            if os.path.isfile(img_path):
                img = cv2.imread(img_path)  # Leitura da imagem usando OpenCV

                # Verificação se a imagem foi lida com sucesso
                if img is not None:
                    # Redimensionamento da imagem
                    img = cv2.resize(img, (64, 64))

                    # Armazenamento da imagem e rótulo correspondente
                    images.append(img)
                    labels.append(label)
                    arq.append(filename)
                else:
                    # Mensagem de erro ao ler a imagem
                    print(f"Erro ao ler a imagem: {img_path}")
            else:
                # Mensagem de erro se o caminho não for válido
                print(f"Caminho inválido: {img_path}")

    return np.array(images), np.array(labels), np.array(arq)

# Carregar imagens de câncer
cancer_images, cancer_labels, com_cancer_arq = Carregar_Imagens(cancer_dir)
cancer_images = cancer_images / 255.0  # Normalização dos valores de pixel

# Carregar imagens sem câncer
sem_cancer_images, sem_cancer_labels, sem_cancer_arq = Carregar_Imagens(sem_cancer_dir)
sem_cancer_images = sem_cancer_images / \
    255.0  # Normalização dos valores de pixel

# Concatenar rótulos para diagnóstico de câncer ou não
labels = np.concatenate((cancer_labels, sem_cancer_labels))
# Convertendo rótulos de texto para valores numéricos (0 e 1)
labels = np.where(labels == 1, 1, 0)

model = Sequential()  # Inicializa um modelo sequencial de camadas

# Adicionando camadas à rede neural
# Camada convolucional 2D com ativação 'relu'
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
# Camada de pooling para redução de dimensionalidade
model.add(MaxPooling2D(pool_size=(2, 2)))
# Outra camada convolucional 2D com ativação 'relu'
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # Nova camada de pooling
# Mais uma camada convolucional 2D com ativação 'relu'
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # Terceira camada de pooling
model.add(Dropout(0.25))  # Aplicação de dropout para prevenir overfitting
# Transforma os dados em um vetor unidimensional para a camada densa
model.add(Flatten())
model.add(Dense(512, activation='relu'))  # Camada densa com ativação 'relu'
model.add(Dropout(0.5))  # Dropout adicional para regularização
# Outra camada densa com ativação 'relu'
model.add(Dense(128, activation='relu'))

# Camada de saída para classificação binária (0 ou 1)
# Camada de saída com ativação 'sigmoid' para classificação binária
model.add(Dense(1, activation='sigmoid'))

# Compilação do modelo
model.compile(loss='binary_crossentropy',  # Usa a entropia cruzada como função de perda
              optimizer='adam',  # Otimizador Adam para ajuste dos pesos
              metrics=['accuracy'])  # Métrica de avaliação: precisão

model.summary()  # Exibe o resumo da arquitetura do modelo criado

# Treinamento do modelo
model.fit(np.concatenate((cancer_images, sem_cancer_images)),
          labels, epochs=20, validation_split=0.2)

# Após o treinamento, Carregamento as imagens do diretório de teste
imagens_nova_classe, rotulos_nova_classe, nova_arq = Carregar_Imagens(diretorio_teste)
imagens_nova_classe = imagens_nova_classe / \
    255.0  # Normalização dos valores de pixel

# Previsões com o modelo treinado usando as imagens do diretório de teste
previsoes_nova_classe = model.predict(imagens_nova_classe)

# Exibição dos resultados das previsões
for i in range(len(imagens_nova_classe)):
    nome_imagem = f"Imagem {i+1}"
    resultado = previsoes_nova_classe[i]
    print(nova_arq[i])
    # Mostra o resultado da predição
    if resultado > 0.5:  # Ajuste o limite conforme necessário
        print(f"{nome_imagem}: Possível câncer")
    else:
        print(f"{nome_imagem}: Não é câncer")

# Avaliação do modelo com os dados de teste
test_loss, test_accuracy = model.evaluate(
    imagens_nova_classe, rotulos_nova_classe)
print(f'A precisão do modelo nos dados de teste é: {test_accuracy:.2%}')
