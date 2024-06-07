import numpy as np
import cv2
import fnmatch
import os
from tensorflow.keras.models import load_model

# Carregar o modelo salvo
model = load_model('modelo_lugares.keras')

# Carregar o dicionário de mapeamento de classes
index_to_class = np.load('class_mapping.npy', allow_pickle=True).item()

def standardize_image(image):
    return (image - np.mean(image, axis=0)) / np.std(image, axis=0)

def prever_localizacao(imagem_path, threshold=0.4):
    """Faz a predição da localização baseada na imagem fornecida.

    Args:
        imagem_path (str): O caminho para a imagem.
        threshold (float): Limiar de confiança para a predição.

    Returns:
        str: Nome da classe ou mensagem de erro se a confiança for baixa.
    """
    # Carregar a imagem usando cv2
    img = cv2.imread(imagem_path)

    # Verificar se a imagem foi carregada corretamente
    if img is None:
        raise ValueError(f"Não foi possível carregar a imagem de {imagem_path}")

    # Redimensionar a imagem para 150x150
    img = cv2.resize(img, (299, 299))

    # Normalizar os valores dos pixels
    img = img / 255.0

    # Padronizar a imagem
    img = standardize_image(img)

    # Expandir as dimensões para (1, 150, 150, 3)
    img = np.expand_dims(img, axis=0)

    # Fazer a predição
    pred = model.predict(img)

    # Obter a probabilidade da classe com maior probabilidade
    max_prob = np.max(pred)

    # Verificar se a probabilidade é menor que o limiar
    if max_prob < threshold:
        return "Local não encontrado na base de dados"

    # Obter o índice da classe com maior probabilidade
    classe_idx = np.argmax(pred)

    # Obter o nome da classe correspondente ao índice
    classe_nome = index_to_class[classe_idx]

    return classe_nome

# Testar a função com uma nova imagem
def percorrer_pasta_e_prever_localizacao(pasta):
    """Percorre todos os arquivos de imagem na pasta especificada e faz predições."""
    for root, _, files in os.walk(pasta):
        for file in files:
            if fnmatch.fnmatch(file, '*.jpg') or fnmatch.fnmatch(file, '*.jpeg') or fnmatch.fnmatch(file, '*.png'):
                caminho_completo = os.path.join(root, file)
                local1 = prever_localizacao(caminho_completo)
                print(f"Localização prevista: {local1} deveria ser: {caminho_completo}")

# Chamar a função com o caminho da pasta desejada
pasta_imagens = './ImagemTeste'
percorrer_pasta_e_prever_localizacao(pasta_imagens)

# Fazer previsões em imagens individuais
local = prever_localizacao('./ImagemTeste/Praca_ucrania.png')
local1 = prever_localizacao('./ImagemTeste/piramedes.jpg')
print(f'deveria ser Praca Ucrania mas é : {local}')
print(f'deveria não encontrar mas é : {local1}')
