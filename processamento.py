"""Algoritmos manuais para imagens estáticas. Não utiliza OpenCV."""

from collections import deque
import math

import numpy as np


# Confere se a imagem é válida, em cinza ou BGR.
def validar_imagem(imagem):
    if not isinstance(imagem, np.ndarray) or imagem.size == 0:
        raise ValueError("A imagem deve ser uma matriz não vazia.")
    if imagem.dtype != np.uint8:
        raise ValueError("A imagem deve conter intensidades uint8 entre 0 e 255.")
    if imagem.ndim != 2 and not (imagem.ndim == 3 and imagem.shape[2] == 3):
        raise ValueError("Use uma imagem em cinza ou com três canais BGR.")


# Aceita máscaras ímpares de 3 a 15.
def validar_tamanho(tamanho):
    if not isinstance(tamanho, int) or tamanho < 3 or tamanho > 15 or tamanho % 2 == 0:
        raise ValueError("O tamanho da máscara deve ser ímpar, entre 3 e 15.")


# Converte as cores para tons de cinza.
def tons_de_cinza(imagem):
    validar_imagem(imagem)
    if imagem.ndim == 2:
        return imagem.copy()
    # Os canais vêm na ordem azul, verde e vermelho.
    azul = imagem[:, :, 0].astype(float)
    verde = imagem[:, :, 1].astype(float)
    vermelho = imagem[:, :, 2].astype(float)
    # O verde tem mais peso por ser mais percebido pelo olho.
    cinza = 0.114 * azul + 0.587 * verde + 0.299 * vermelho
    return np.clip(np.rint(cinza), 0, 255).astype(np.uint8)


# Inverte as intensidades da imagem.
def negativo(imagem):
    validar_imagem(imagem)
    return 255 - imagem


# Conta os pixels de cada intensidade, de 0 a 255.
def histograma(imagem):
    cinza = tons_de_cinza(imagem)
    contagem = np.zeros(256, dtype=np.int64)
    for intensidade in cinza.flat:
        contagem[int(intensidade)] += 1
    return contagem


# Escolhe o limiar automaticamente e gera preto e branco.
def otsu(imagem):
    cinza = tons_de_cinza(imagem)
    contagem = histograma(cinza)
    probabilidades = contagem / cinza.size
    # Calcula a intensidade média da imagem.
    media_total = 0.0
    for intensidade in range(256):
        media_total += intensidade * probabilidades[intensidade]

    peso_fundo = 0.0
    soma_fundo = 0.0
    maior_variancia = -1.0
    # Se a imagem for uniforme, usa a própria intensidade.
    melhor_limiar = int(cinza[0, 0])
    # Testa cada limiar separando dois grupos de intensidades.
    for limiar in range(256):
        peso_fundo += probabilidades[limiar]
        soma_fundo += limiar * probabilidades[limiar]
        peso_objeto = 1.0 - peso_fundo
        if peso_fundo <= 0 or peso_objeto <= 1e-12:
            continue  # Pula grupos vazios para evitar divisão por zero.
        media_fundo = soma_fundo / peso_fundo
        media_objeto = (media_total - soma_fundo) / peso_objeto
        # Guarda o limiar com maior separação entre os grupos.
        variancia = peso_fundo * peso_objeto * (media_fundo - media_objeto) ** 2
        if variancia > maior_variancia:
            maior_variancia = variancia
            melhor_limiar = limiar

    # Acima do limiar fica branco; o restante fica preto.
    binaria = np.zeros_like(cinza)
    binaria[cinza > melhor_limiar] = 255
    return binaria, melhor_limiar


# Suaviza cada pixel com a média da vizinhança.
def filtro_media(imagem, tamanho=3):
    validar_imagem(imagem)
    validar_tamanho(tamanho)
    raio = tamanho // 2
    margem = ((raio, raio), (raio, raio))
    if imagem.ndim == 3:
        margem += ((0, 0),)
    # Repete as bordas para a máscara caber em todos os pixels.
    preenchida = np.pad(imagem, margem, mode="edge")
    resultado = np.empty_like(imagem)
    altura, largura = imagem.shape[:2]
    for y in range(altura):
        for x in range(largura):
            vizinhanca = preenchida[y:y + tamanho, x:x + tamanho]
            if imagem.ndim == 2:
                media = np.sum(vizinhanca, dtype=float) / (tamanho * tamanho)
            else:
                # Calcula a média de cada canal separadamente.
                media = np.sum(vizinhanca, axis=(0, 1), dtype=float) / (tamanho * tamanho)
            resultado[y, x] = np.clip(np.rint(media), 0, 255)
    return resultado


# Reduz ruídos usando a mediana da vizinhança.
def filtro_mediana(imagem, tamanho=3):
    validar_imagem(imagem)
    validar_tamanho(tamanho)
    raio = tamanho // 2
    margem = ((raio, raio), (raio, raio))
    if imagem.ndim == 3:
        margem += ((0, 0),)
    preenchida = np.pad(imagem, margem, mode="edge")
    resultado = np.empty_like(imagem)
    # A mediana é o valor central depois de ordenar.
    meio = tamanho * tamanho // 2
    for y in range(imagem.shape[0]):
        for x in range(imagem.shape[1]):
            vizinhanca = preenchida[y:y + tamanho, x:x + tamanho]
            if imagem.ndim == 2:
                valores = sorted(vizinhanca.flatten().tolist())
                resultado[y, x] = valores[meio]
            else:
                # Trata cada canal de cor separadamente.
                for canal in range(3):
                    valores = sorted(vizinhanca[:, :, canal].flatten().tolist())
                    resultado[y, x, canal] = valores[meio]
    return resultado


# Percorre a imagem aplicando os pesos da máscara.
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

# Afina as bordas mantendo os maiores valores locais.
def suprimir_nao_maximos(magnitude, direcao):
    altura, largura = magnitude.shape
    resultado = np.zeros_like(magnitude)
    # Compara com dois vizinhos na direção do gradiente.
    for y in range(1, altura - 1):
        for x in range(1, largura - 1):
            # O ângulo define quais vizinhos comparar.
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


# Mantém bordas fortes e as fracas ligadas a elas.
def histerese(magnitude, limiar_inferior, limiar_superior):
    fortes = magnitude >= limiar_superior
    candidatas = magnitude >= limiar_inferior
    resultado = np.zeros(magnitude.shape, dtype=np.uint8)
    # Começa a busca pelos pixels de borda forte.
    fila = deque()
    for y in range(magnitude.shape[0]):
        for x in range(magnitude.shape[1]):
            if fortes[y, x]:
                resultado[y, x] = 255
                fila.append((y, x))
    # Segue as conexões pelos oito vizinhos de cada pixel.
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

# Detecta bordas com as etapas do Canny.
def canny(imagem, limiar_inferior=40, limiar_superior=100):
    if not (0 < limiar_inferior < limiar_superior <= 1500):
        raise ValueError("No Canny, use 0 < limiar inferior < superior <= 1500.")
    cinza = tons_de_cinza(imagem)
    # A máscara gaussiana suaviza a imagem antes de buscar bordas.
    gaussiana = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4],
                         [6, 24, 36, 24, 6], [4, 16, 24, 16, 4],
                         [1, 4, 6, 4, 1]], dtype=float) / 256
    suavizada = aplicar_mascara(cinza, gaussiana)
    # Sobel mede a variação de intensidade em cada direção.
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
    gradiente_x = aplicar_mascara(suavizada, sobel_x)
    gradiente_y = aplicar_mascara(suavizada, sobel_y)
    # Calcula a força e a direção das bordas.
    magnitude = np.sqrt(gradiente_x ** 2 + gradiente_y ** 2)
    direcao = np.degrees(np.arctan2(gradiente_y, gradiente_x))
    # Afina as bordas e mantém as conexões com as fortes.
    finas = suprimir_nao_maximos(magnitude, direcao)
    return histerese(finas, limiar_inferior, limiar_superior)

# Confere se a imagem tem apenas preto (0) e branco (255).
def exigir_binaria(imagem):
    validar_imagem(imagem)
    if imagem.ndim != 2 or not np.all((imagem == 0) | (imagem == 255)):
        raise ValueError("Esta operação exige imagem binária (0 e 255). Aplique Otsu ou Canny primeiro.")

# Encolhe as regiões brancas.
def erosao(imagem, tamanho=3):
    exigir_binaria(imagem)
    validar_tamanho(tamanho)
    preenchida = np.pad(imagem, tamanho // 2, mode="constant", constant_values=0)
    resultado = np.zeros_like(imagem)
    for y in range(imagem.shape[0]):
        for x in range(imagem.shape[1]):
            vizinhanca = preenchida[y:y + tamanho, x:x + tamanho]
            # Só mantém branco se toda a vizinhança for branca.
            if np.all(vizinhanca == 255):
                resultado[y, x] = 255
    return resultado


# Expande as regiões brancas.
def dilatacao(imagem, tamanho=3):
    exigir_binaria(imagem)
    validar_tamanho(tamanho)
    preenchida = np.pad(imagem, tamanho // 2, mode="constant", constant_values=0)
    resultado = np.zeros_like(imagem)
    for y in range(imagem.shape[0]):
        for x in range(imagem.shape[1]):
            vizinhanca = preenchida[y:y + tamanho, x:x + tamanho]
            # Basta um pixel branco na vizinhança para expandir.
            if np.any(vizinhanca == 255):
                resultado[y, x] = 255
    return resultado


# Erosão e depois dilatação: remove pequenas regiões brancas.
def abertura(imagem, tamanho=3):
    return dilatacao(erosao(imagem, tamanho), tamanho)


# Dilatação e depois erosão: preenche pequenos buracos.
def fechamento(imagem, tamanho=3):
    return erosao(dilatacao(imagem, tamanho), tamanho)


# Agrupa os pixels brancos conectados em objetos.
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
            # Cada objeto novo recebe um número próprio.
            rotulo = len(objetos) + 1
            fila = deque([(y, x)])
            rotulos[y, x] = rotulo
            pixels = []
            # A fila percorre os pixels ligados pelos oito vizinhos.
            while fila:
                atual_y, atual_x = fila.popleft()
                pixels.append((atual_y, atual_x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = atual_y + dy, atual_x + dx
                        if 0 <= ny < altura and 0 <= nx < largura:
                            if imagem[ny, nx] == 255 and rotulos[ny, nx] == 0:
                                # Marca antes de colocar na fila para não repetir.
                                rotulos[ny, nx] = rotulo
                                fila.append((ny, nx))
            objetos.append(pixels)
    return rotulos, objetos


# Calcula área, perímetro e diâmetro de cada objeto.
def medir_objetos(rotulos, objetos):
    medidas = []
    altura, largura = rotulos.shape
    for rotulo, pixels in enumerate(objetos, start=1):
        perimetro = 0
        borda = []
        for y, x in pixels:
            esta_na_borda = False
            # Cada lado exposto do pixel soma um ao perímetro.
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= altura or nx < 0 or nx >= largura or rotulos[ny, nx] != rotulo:
                    perimetro += 1
                    esta_na_borda = True
            if esta_na_borda:
                borda.append((y, x))

        # O diâmetro é a maior distância entre centros de pixels da borda.
        maior_distancia_quadrada = 0
        for i in range(len(borda)):
            y1, x1 = borda[i]
            for j in range(i + 1, len(borda)):
                y2, x2 = borda[j]
                distancia_quadrada = (y1 - y2) ** 2 + (x1 - x2) ** 2
                if distancia_quadrada > maior_distancia_quadrada:
                    maior_distancia_quadrada = distancia_quadrada
        # A área é a quantidade de pixels do objeto.
        medidas.append({"objeto": rotulo, "area": len(pixels), "perimetro": perimetro,
                        "diametro": math.sqrt(maior_distancia_quadrada)})
    return medidas


# Usa o número de cada objeto para definir sua cor.
def colorir_componentes(rotulos):
    altura, largura = rotulos.shape
    colorida = np.zeros((altura, largura, 3), dtype=np.uint8)
    for y in range(altura):
        for x in range(largura):
            rotulo = int(rotulos[y, x])
            if rotulo:
                # O mesmo rótulo sempre gera a mesma cor BGR.
                colorida[y, x] = (50 + (rotulo * 67) % 206,
                                  50 + (rotulo * 113) % 206,
                                  50 + (rotulo * 173) % 206)
    return colorida
