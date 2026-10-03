"""Algoritmos manuais para imagens estáticas. Não utiliza OpenCV."""

from collections import deque
import math

import numpy as np


def validar_imagem(imagem):
    if not isinstance(imagem, np.ndarray) or imagem.size == 0:
        raise ValueError("A imagem deve ser uma matriz não vazia.")
    if imagem.dtype != np.uint8:
        raise ValueError("A imagem deve conter intensidades uint8 entre 0 e 255.")
    if imagem.ndim != 2 and not (imagem.ndim == 3 and imagem.shape[2] == 3):
        raise ValueError("Use uma imagem em cinza ou com três canais BGR.")


def validar_tamanho(tamanho):
    if not isinstance(tamanho, int) or tamanho < 3 or tamanho > 15 or tamanho % 2 == 0:
        raise ValueError("O tamanho da máscara deve ser ímpar, entre 3 e 15.")


def tons_de_cinza(imagem):
    validar_imagem(imagem)
    if imagem.ndim == 2:
        return imagem.copy()
    # O arquivo aberto pelo OpenCV está em BGR, e não em RGB.
    azul = imagem[:, :, 0].astype(float)
    verde = imagem[:, :, 1].astype(float)
    vermelho = imagem[:, :, 2].astype(float)
    cinza = 0.114 * azul + 0.587 * verde + 0.299 * vermelho
    return np.clip(np.rint(cinza), 0, 255).astype(np.uint8)


def negativo(imagem):
    validar_imagem(imagem)
    return 255 - imagem


def histograma(imagem):
    cinza = tons_de_cinza(imagem)
    contagem = np.zeros(256, dtype=np.int64)
    for intensidade in cinza.flat:
        contagem[int(intensidade)] += 1
    return contagem


def otsu(imagem):
    cinza = tons_de_cinza(imagem)
    contagem = histograma(cinza)
    probabilidades = contagem / cinza.size
    media_total = 0.0
    for intensidade in range(256):
        media_total += intensidade * probabilidades[intensidade]

    peso_fundo = 0.0
    soma_fundo = 0.0
    maior_variancia = -1.0
    # Uma imagem uniforme não tem duas classes. O limiar será sua intensidade.
    melhor_limiar = int(cinza[0, 0])
    for limiar in range(256):
        peso_fundo += probabilidades[limiar]
        soma_fundo += limiar * probabilidades[limiar]
        peso_objeto = 1.0 - peso_fundo
        if peso_fundo <= 0 or peso_objeto <= 1e-12:
            continue  # Evita dividir pelo peso de uma classe vazia.
        media_fundo = soma_fundo / peso_fundo
        media_objeto = (media_total - soma_fundo) / peso_objeto
        variancia = peso_fundo * peso_objeto * (media_fundo - media_objeto) ** 2
        if variancia > maior_variancia:
            maior_variancia = variancia
            melhor_limiar = limiar

    binaria = np.zeros_like(cinza)
    binaria[cinza > melhor_limiar] = 255
    return binaria, melhor_limiar


def filtro_media(imagem, tamanho=3):
    validar_imagem(imagem)
    validar_tamanho(tamanho)
    raio = tamanho // 2
    margem = ((raio, raio), (raio, raio))
    if imagem.ndim == 3:
        margem += ((0, 0),)
    # Repete o pixel mais próximo nas bordas para manter o tamanho da imagem.
    preenchida = np.pad(imagem, margem, mode="edge")
    resultado = np.empty_like(imagem)
    altura, largura = imagem.shape[:2]
    for y in range(altura):
        for x in range(largura):
            vizinhanca = preenchida[y:y + tamanho, x:x + tamanho]
            if imagem.ndim == 2:
                media = np.sum(vizinhanca, dtype=float) / (tamanho * tamanho)
            else:
                media = np.sum(vizinhanca, axis=(0, 1), dtype=float) / (tamanho * tamanho)
            resultado[y, x] = np.clip(np.rint(media), 0, 255)
    return resultado


def filtro_mediana(imagem, tamanho=3):
    validar_imagem(imagem)
    validar_tamanho(tamanho)
    raio = tamanho // 2
    margem = ((raio, raio), (raio, raio))
    if imagem.ndim == 3:
        margem += ((0, 0),)
    preenchida = np.pad(imagem, margem, mode="edge")
    resultado = np.empty_like(imagem)
    meio = tamanho * tamanho // 2
    for y in range(imagem.shape[0]):
        for x in range(imagem.shape[1]):
            vizinhanca = preenchida[y:y + tamanho, x:x + tamanho]
            if imagem.ndim == 2:
                valores = sorted(vizinhanca.flatten().tolist())
                resultado[y, x] = valores[meio]
            else:
                for canal in range(3):
                    valores = sorted(vizinhanca[:, :, canal].flatten().tolist())
                    resultado[y, x, canal] = valores[meio]
    return resultado


def aplicar_mascara(cinza, mascara):
    """Soma os produtos entre uma vizinhança e a máscara (correlação)."""
    tamanho = mascara.shape[0]
    raio = tamanho // 2
    preenchida = np.pad(cinza.astype(float), raio, mode="edge")
    resultado = np.zeros(cinza.shape, dtype=float)
    for y in range(cinza.shape[0]):
        for x in range(cinza.shape[1]):
            vizinhanca = preenchida[y:y + tamanho, x:x + tamanho]
            resultado[y, x] = np.sum(vizinhanca * mascara)
    return resultado


def suprimir_nao_maximos(magnitude, direcao):
    altura, largura = magnitude.shape
    resultado = np.zeros_like(magnitude)
    # Compara cada magnitude com os dois vizinhos na direção do gradiente.
    for y in range(1, altura - 1):
        for x in range(1, largura - 1):
            angulo = direcao[y, x] % 180
            if angulo < 22.5 or angulo >= 157.5:
                primeiro, segundo = magnitude[y, x - 1], magnitude[y, x + 1]
            elif angulo < 67.5:
                primeiro, segundo = magnitude[y - 1, x - 1], magnitude[y + 1, x + 1]
            elif angulo < 112.5:
                primeiro, segundo = magnitude[y - 1, x], magnitude[y + 1, x]
            else:
                primeiro, segundo = magnitude[y - 1, x + 1], magnitude[y + 1, x - 1]
            if magnitude[y, x] >= primeiro and magnitude[y, x] >= segundo:
                resultado[y, x] = magnitude[y, x]
    return resultado


def histerese(magnitude, limiar_inferior, limiar_superior):
    fortes = magnitude >= limiar_superior
    candidatas = magnitude >= limiar_inferior
    resultado = np.zeros(magnitude.shape, dtype=np.uint8)
    fila = deque()
    for y in range(magnitude.shape[0]):
        for x in range(magnitude.shape[1]):
            if fortes[y, x]:
                resultado[y, x] = 255
                fila.append((y, x))
    # Bordas fracas só sobrevivem se estiverem conectadas a uma borda forte.
    while fila:
        y, x = fila.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < magnitude.shape[0] and 0 <= nx < magnitude.shape[1]:
                    if candidatas[ny, nx] and resultado[ny, nx] == 0:
                        resultado[ny, nx] = 255
                        fila.append((ny, nx))
    return resultado


def canny(imagem, limiar_inferior=40, limiar_superior=100):
    if not (0 < limiar_inferior < limiar_superior <= 1500):
        raise ValueError("No Canny, use 0 < limiar inferior < superior <= 1500.")
    cinza = tons_de_cinza(imagem)
    # Suavização gaussiana 5x5: pesos maiores perto do centro, soma dos pesos = 256.
    gaussiana = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4],
                         [6, 24, 36, 24, 6], [4, 16, 24, 16, 4],
                         [1, 4, 6, 4, 1]], dtype=float) / 256
    suavizada = aplicar_mascara(cinza, gaussiana)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
    gradiente_x = aplicar_mascara(suavizada, sobel_x)
    gradiente_y = aplicar_mascara(suavizada, sobel_y)
    magnitude = np.sqrt(gradiente_x ** 2 + gradiente_y ** 2)
    direcao = np.degrees(np.arctan2(gradiente_y, gradiente_x))
    finas = suprimir_nao_maximos(magnitude, direcao)
    return histerese(finas, limiar_inferior, limiar_superior)


def exigir_binaria(imagem):
    validar_imagem(imagem)
    if imagem.ndim != 2 or not np.all((imagem == 0) | (imagem == 255)):
        raise ValueError("Esta operação exige imagem binária (0 e 255). Aplique Otsu ou Canny primeiro.")


def erosao(imagem, tamanho=3):
    exigir_binaria(imagem)
    validar_tamanho(tamanho)
    preenchida = np.pad(imagem, tamanho // 2, mode="constant", constant_values=0)
    resultado = np.zeros_like(imagem)
    for y in range(imagem.shape[0]):
        for x in range(imagem.shape[1]):
            vizinhanca = preenchida[y:y + tamanho, x:x + tamanho]
            if np.all(vizinhanca == 255):
                resultado[y, x] = 255
    return resultado


def dilatacao(imagem, tamanho=3):
    exigir_binaria(imagem)
    validar_tamanho(tamanho)
    preenchida = np.pad(imagem, tamanho // 2, mode="constant", constant_values=0)
    resultado = np.zeros_like(imagem)
    for y in range(imagem.shape[0]):
        for x in range(imagem.shape[1]):
            vizinhanca = preenchida[y:y + tamanho, x:x + tamanho]
            if np.any(vizinhanca == 255):
                resultado[y, x] = 255
    return resultado


def abertura(imagem, tamanho=3):
    return dilatacao(erosao(imagem, tamanho), tamanho)


def fechamento(imagem, tamanho=3):
    return erosao(dilatacao(imagem, tamanho), tamanho)


def componentes_conexos(imagem):
    """Retorna matriz de rótulos e uma lista com os pixels de cada objeto branco."""
    exigir_binaria(imagem)
    altura, largura = imagem.shape
    rotulos = np.zeros(imagem.shape, dtype=np.int32)
    objetos = []
    for y in range(altura):
        for x in range(largura):
            if imagem[y, x] != 255 or rotulos[y, x] != 0:
                continue
            rotulo = len(objetos) + 1
            fila = deque([(y, x)])
            rotulos[y, x] = rotulo
            pixels = []
            while fila:
                atual_y, atual_x = fila.popleft()
                pixels.append((atual_y, atual_x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = atual_y + dy, atual_x + dx
                        if 0 <= ny < altura and 0 <= nx < largura:
                            if imagem[ny, nx] == 255 and rotulos[ny, nx] == 0:
                                # Marca antes de enfileirar para não repetir o mesmo pixel.
                                rotulos[ny, nx] = rotulo
                                fila.append((ny, nx))
            objetos.append(pixels)
    return rotulos, objetos


def medir_objetos(rotulos, objetos):
    medidas = []
    altura, largura = rotulos.shape
    for rotulo, pixels in enumerate(objetos, start=1):
        perimetro = 0
        borda = []
        for y, x in pixels:
            esta_na_borda = False
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= altura or nx < 0 or nx >= largura or rotulos[ny, nx] != rotulo:
                    perimetro += 1
                    esta_na_borda = True
            if esta_na_borda:
                borda.append((y, x))

        # Maior distância entre centros de pixels de borda. Compara todos os pares.
        maior_distancia_quadrada = 0
        for i in range(len(borda)):
            y1, x1 = borda[i]
            for j in range(i + 1, len(borda)):
                y2, x2 = borda[j]
                distancia_quadrada = (y1 - y2) ** 2 + (x1 - x2) ** 2
                if distancia_quadrada > maior_distancia_quadrada:
                    maior_distancia_quadrada = distancia_quadrada
        medidas.append({"objeto": rotulo, "area": len(pixels), "perimetro": perimetro,
                        "diametro": math.sqrt(maior_distancia_quadrada)})
    return medidas


def colorir_componentes(rotulos):
    altura, largura = rotulos.shape
    colorida = np.zeros((altura, largura, 3), dtype=np.uint8)
    for y in range(altura):
        for x in range(largura):
            rotulo = int(rotulos[y, x])
            if rotulo:
                # Cores BGR reproduzíveis; o fundo permanece preto.
                colorida[y, x] = (50 + (rotulo * 67) % 206,
                                  50 + (rotulo * 113) % 206,
                                  50 + (rotulo * 173) % 206)
    return colorida
