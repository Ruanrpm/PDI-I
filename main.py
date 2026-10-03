"""Interface Tkinter do trabalho de PDI. Execute: python main.py."""

import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import cv2
import numpy as np
from PIL import Image, ImageTk
import pygame

import processamento as pdi


MODOS_VIDEO = ("Original", "Cinza", "Negativo", "Otsu", "Média", "Mediana",
               "Canny", "Erosão", "Dilatação", "Abertura", "Fechamento")


def ler_imagem(caminho):
    # imdecode/fromfile também aceita caminhos com acentos no Windows.
    dados = np.fromfile(caminho, dtype=np.uint8)
    if dados.size == 0:
        raise ValueError("O arquivo está vazio.")
    imagem = cv2.imdecode(dados, cv2.IMREAD_COLOR)
    if imagem is None:
        raise ValueError("O arquivo selecionado não é uma imagem válida.")
    return imagem


def salvar_imagem(caminho, imagem):
    extensao = os.path.splitext(caminho)[1].lower()
    if extensao not in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        raise ValueError("Escolha PNG, JPG, BMP ou TIFF para salvar.")
    sucesso, dados = cv2.imencode(extensao, imagem)
    if not sucesso:
        raise ValueError("Não foi possível codificar a imagem.")
    dados.tofile(caminho)


def processar_video(frame, modo, tamanho=3, inferior=40, superior=100, estrutura=3):
    """Somente no vídeo usamos as operações prontas do OpenCV."""
    if modo == "Original":
        return frame.copy()
    if modo == "Negativo":
        return 255 - frame
    if modo == "Média":
        return cv2.blur(frame, (tamanho, tamanho))
    if modo == "Mediana":
        return cv2.medianBlur(frame, tamanho)
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if modo == "Cinza":
        return cinza
    if modo == "Canny":
        return cv2.Canny(cinza, inferior, superior)
    _, binaria = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elemento = np.ones((estrutura, estrutura), dtype=np.uint8)
    if modo == "Otsu":
        return binaria
    if modo == "Erosão":
        return cv2.erode(binaria, elemento)
    if modo == "Dilatação":
        return cv2.dilate(binaria, elemento)
    if modo == "Abertura":
        return cv2.morphologyEx(binaria, cv2.MORPH_OPEN, elemento)
    if modo == "Fechamento":
        return cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, elemento)
    raise ValueError("Modo de vídeo desconhecido.")


def preparar_rastreamento(frame, regiao):
    x, y, largura, altura = regiao
    recorte = frame[y:y + altura, x:x + largura]
    if largura < 5 or altura < 5 or recorte.size == 0:
        raise ValueError("Selecione uma região de pelo menos 5 x 5 pixels.")
    hsv = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV)
    # Desconsidera pixels quase sem cor ou muito escuros, que não têm matiz confiável.
    mascara = cv2.inRange(hsv, np.array([0, 40, 30]), np.array([179, 255, 255]))
    if cv2.countNonZero(mascara) == 0:
        raise ValueError("Escolha um objeto com cor visível; a região está escura ou sem saturação.")
    hist = cv2.calcHist([hsv], [0], mascara, [180], [0, 180])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist


def rastrear_frame(frame, hist, janela, referencia=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    projecao = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
    mascara = cv2.inRange(hsv, np.array([0, 40, 30]), np.array([179, 255, 255]))
    projecao = cv2.bitwise_and(projecao, mascara)
    x, y, largura, altura = janela
    recorte = projecao[y:y + altura, x:x + largura]
    if largura <= 0 or altura <= 0 or np.count_nonzero(recorte) < max(5, recorte.size * 0.03):
        return None, janela
    criterio = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    caixa, nova_janela = cv2.CamShift(projecao, janela, criterio)
    if referencia is not None:
        # Confere a aparência perto da nova posição, evitando seguir apenas uma cor do fundo.
        x, y, largura, altura = nova_janela
        margem_x = max(largura, referencia.shape[1]) // 2
        margem_y = max(altura, referencia.shape[0]) // 2
        esquerda = max(0, x - margem_x)
        superior = max(0, y - margem_y)
        direita = min(frame.shape[1], x + largura + margem_x)
        inferior = min(frame.shape[0], y + altura + margem_y)
        regiao = frame[superior:inferior, esquerda:direita]
        if regiao.size == 0 or reencontrar_objeto(regiao, referencia, hist) is None:
            return None, nova_janela
    return caixa, nova_janela


def reencontrar_objeto(frame, referencia, hist, limiar=0.65):
    """Busca a região selecionada novamente no vídeo, sem apagar o modelo original."""
    if referencia is None or frame.size == 0:
        return None
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    modelo = cv2.cvtColor(referencia, cv2.COLOR_BGR2GRAY)
    if float(np.std(modelo)) >= 2:
        melhor_similaridade = -1.0
        melhor_regiao = None
        # Três escalas simples toleram pequenas mudanças na distância da câmera.
        for escala in (0.75, 1.0, 1.25):
            largura = max(5, round(modelo.shape[1] * escala))
            altura = max(5, round(modelo.shape[0] * escala))
            if altura > cinza.shape[0] or largura > cinza.shape[1]:
                continue
            redimensionado = cv2.resize(modelo, (largura, altura))
            if float(np.std(redimensionado)) < 1:
                continue
            resultado = cv2.matchTemplate(cinza, redimensionado, cv2.TM_CCOEFF_NORMED)
            _, similaridade, _, (x, y) = cv2.minMaxLoc(resultado)
            if similaridade > melhor_similaridade:
                melhor_similaridade = similaridade
                melhor_regiao = (x, y, largura, altura)
        return melhor_regiao if melhor_similaridade >= limiar else None

    # Um objeto de cor uniforme não tem detalhes para template matching.
    # Nesse caso, busca regiões da cor aprendida. Estas funções prontas são só do VÍDEO.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    projecao = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
    mascara = cv2.inRange(hsv, np.array([0, 40, 30]), np.array([179, 255, 255]))
    projecao = cv2.bitwise_and(projecao, mascara)
    _, binaria = cv2.threshold(projecao, 80, 255, cv2.THRESH_BINARY)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    quantidade, _, estatisticas, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    area_modelo = referencia.shape[0] * referencia.shape[1]
    maior_area = 0
    melhor_regiao = None
    for rotulo in range(1, quantidade):
        x, y, largura, altura, area = estatisticas[rotulo]
        if max(10, area_modelo * 0.2) <= area <= area_modelo * 4 and area > maior_area:
            maior_area = int(area)
            melhor_regiao = (int(x), int(y), int(largura), int(altura))
    return melhor_regiao


def preparar_referencia(imagem):
    referencia = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    if min(referencia.shape) < 5:
        raise ValueError("A referência deve ter pelo menos 5 x 5 pixels.")
    # Fotos de arquivo podem ter milhares de pixels. Um modelo pequeno permite
    # comparar vários tamanhos no vídeo sem exigir a resolução original da foto.
    altura, largura = referencia.shape
    if max(altura, largura) > 160:
        escala = 160 / max(altura, largura)
        novo_tamanho = (max(1, round(largura * escala)), max(1, round(altura * escala)))
        referencia = cv2.resize(referencia, novo_tamanho, interpolation=cv2.INTER_AREA)
    if min(referencia.shape) < 5:
        raise ValueError("O recorte é muito estreito. Escolha uma região que contenha o objeto.")
    # Correlação normalizada não é adequada para um modelo de intensidade uniforme.
    if float(np.std(referencia)) < 1:
        raise ValueError("A referência precisa ter detalhes/contraste. Recorte o objeto desejado.")
    return referencia


def localizar_referencia(frame, referencia):
    """Template matching em tamanhos diferentes, somente para identificação no vídeo."""
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if float(np.std(referencia)) < 1:
        raise ValueError("A referência precisa ter detalhes/contraste. Recorte o objeto desejado.")
    melhor_similaridade = -1.0
    melhor_regiao = None
    # Aproximar/afastar da câmera muda o tamanho do objeto. Testamos alguns
    # tamanhos mantendo sua proporção e guardamos a maior correlação.
    for escala in (0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0):
        largura = round(referencia.shape[1] * escala)
        altura = round(referencia.shape[0] * escala)
        if min(largura, altura) < 5 or altura > cinza.shape[0] or largura > cinza.shape[1]:
            continue
        modelo = cv2.resize(referencia, (largura, altura))
        if float(np.std(modelo)) < 1:
            continue
        similaridades = cv2.matchTemplate(cinza, modelo, cv2.TM_CCOEFF_NORMED)
        _, similaridade, _, (x, y) = cv2.minMaxLoc(similaridades)
        if similaridade > melhor_similaridade:
            melhor_similaridade = float(similaridade)
            melhor_regiao = (x, y, largura, altura)
    if melhor_regiao is None:
        raise ValueError("Nenhum tamanho da referência cabe no vídeo. Use um recorte menor do objeto.")
    return melhor_similaridade, melhor_regiao


def atualizar_presenca(encontrado, presente, ausencias, limite_ausencias=5):
    """Dispara só na entrada, após o número definido de frames ausentes."""
    if encontrado:
        disparar = not presente
        return True, 0, disparar
    ausencias += 1
    if ausencias >= limite_ausencias:
        presente = False
    return presente, ausencias, False


class Aplicacao:
    """Uma única classe guarda os widgets e o estado da interface."""

    def __init__(self, raiz):
        self.raiz = raiz
        raiz.title("Trabalho de Processamento Digital de Imagens")
        largura_tela = raiz.winfo_screenwidth()
        altura_tela = raiz.winfo_screenheight()
        largura_janela = min(1180, largura_tela - 80)
        altura_janela = min(780, altura_tela - 100)
        posicao_x = max(0, (largura_tela - largura_janela) // 2)
        posicao_y = max(0, (altura_tela - altura_janela) // 2 - 20)
        raiz.geometry(f"{largura_janela}x{altura_janela}+{posicao_x}+{posicao_y}")
        raiz.minsize(min(920, largura_janela), min(600, altura_janela))
        self.original = None
        self.atual = None
        self.visualizacao = None
        self.ocupado = False
        self.fila_resultados = queue.Queue()
        self.botoes_imagem = []
        self.captura = None
        self.frame_atual = None
        self.agendamento_camera = None
        self.hist_rastreamento = None
        self.janela_rastreamento = None
        self.referencia_rastreamento = None
        self.rastreamento_perdido = False
        self.selecionando = False
        self.inicio_selecao = None
        self.frame_selecao = None
        self.transformacao_camera = None
        self.referencia = None
        self.referencia_ativa = False
        self.caminho_musica = None
        self.objeto_presente = False
        self.frames_ausentes = 0
        self.ultimo_disparo = -float("inf")
        self.musica_pendente = False
        self.musica_pausada = False
        # A busca da referência pode funcionar sem música; este estado habilita o áudio.
        self.identificacao_ativa = False
        self.fonte_identificacao = None

        self.status = tk.StringVar(value="Carregue uma imagem ou abra a câmera.")
        self.tamanho_filtro = tk.StringVar(value="3")
        self.tamanho_estrutura = tk.StringVar(value="3")
        self.limiar_inferior = tk.StringVar(value="40")
        self.limiar_superior = tk.StringVar(value="100")
        self.indice_camera = tk.StringVar(value="0")
        self.modo_video = tk.StringVar(value="Original")
        self.limiar_similaridade = tk.StringVar(value="0.80")
        self.info_camera = tk.StringVar(value="Câmera fechada.")
        self.info_referencia = tk.StringVar(value="Nenhuma referência selecionada.")
        self.info_musica = tk.StringVar(value="Nenhuma música selecionada.")
        self.info_identificacao = tk.StringVar(value="Selecione um objeto e uma música para ativar o áudio.")
        self.criar_interface()
        self.agendamento_resultado = self.raiz.after(80, self.receber_resultado)
        self.raiz.protocol("WM_DELETE_WINDOW", self.encerrar)

    def criar_interface(self):
        # Os controles têm espaço reservado; somente as imagens expandem.
        self.raiz.columnconfigure(0, weight=1)
        self.raiz.rowconfigure(2, weight=1)
        topo = ttk.Frame(self.raiz, padding=8)
        topo.grid(row=0, column=0, sticky="ew")
        for texto, comando in (("Carregar imagem", self.carregar_imagem),
                               ("Restaurar original", self.restaurar),
                               ("Salvar resultado", self.salvar)):
            botao = ttk.Button(topo, text=texto, command=comando)
            botao.pack(side="left", padx=4)
            if texto == "Carregar imagem":
                self.botao_carregar_imagem = botao
            else:
                self.botoes_imagem.append(botao)
        ttk.Button(topo, text="Fechar câmera", command=self.fechar_camera).pack(side="right", padx=4)
        self.botao_camera_topo = ttk.Button(topo, text="Abrir câmera", command=self.abrir_camera)
        self.botao_camera_topo.pack(side="right", padx=4)

        visualizacoes = ttk.Frame(self.raiz, padding=(8, 0))
        visualizacoes.grid(row=2, column=0, sticky="nsew")
        visualizacoes.columnconfigure(0, weight=1)
        visualizacoes.columnconfigure(1, weight=1)
        visualizacoes.rowconfigure(0, weight=1)
        esquerda = ttk.LabelFrame(visualizacoes, text="Imagem original / câmera original")
        direita = ttk.LabelFrame(visualizacoes, text="Resultado / câmera processada")
        esquerda.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        direita.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        self.canvas_original = tk.Canvas(esquerda, bg="#252525", highlightthickness=0, height=280)
        self.canvas_resultado = tk.Canvas(direita, bg="#252525", highlightthickness=0, height=280)
        self.canvas_original.pack(fill="both", expand=True)
        self.canvas_resultado.pack(fill="both", expand=True)
        self.canvas_original.bind("<Configure>", lambda evento: self.redesenhar_imagens())
        self.canvas_resultado.bind("<Configure>", lambda evento: self.redesenhar_imagens())
        self.canvas_original.bind("<ButtonPress-1>", self.iniciar_selecao)
        self.canvas_original.bind("<B1-Motion>", self.mover_selecao)
        self.canvas_original.bind("<ButtonRelease-1>", self.terminar_selecao)

        self.abas = ttk.Notebook(self.raiz)
        self.abas.grid(row=1, column=0, sticky="ew", padx=8, pady=8)
        imagens = ttk.Frame(self.abas, padding=8)
        video = ttk.Frame(self.abas, padding=8)
        self.abas.add(imagens, text="Imagem estática — algoritmos manuais")
        self.abas.add(video, text="Vídeo, rastreamento e música")
        self.abas.bind("<<NotebookTabChanged>>", self.trocar_aba)

        conversoes = ttk.LabelFrame(imagens, text="Conversões", padding=6)
        filtros = ttk.LabelFrame(imagens, text="Filtros", padding=6)
        morfologia = ttk.LabelFrame(imagens, text="Morfologia binária", padding=6)
        analise = ttk.LabelFrame(imagens, text="Análise", padding=6)
        for coluna, secao in enumerate((conversoes, filtros, morfologia, analise)):
            secao.grid(row=0, column=coluna, sticky="nsew", padx=4)
            imagens.columnconfigure(coluna, weight=1)
        for linha, nome in enumerate(("Cinza", "Negativo", "Otsu")):
            self.botao_operacao(conversoes, nome, linha)
        ttk.Label(filtros, text="Máscara (lado):").grid(row=0, column=0)
        ttk.Combobox(filtros, textvariable=self.tamanho_filtro, values=(3, 5, 7), state="readonly", width=5).grid(row=0, column=1)
        for linha, nome in enumerate(("Média", "Mediana", "Canny"), start=1):
            self.botao_operacao(filtros, nome, linha)
        ttk.Label(filtros, text="Canny inferior / superior:").grid(row=4, column=0, columnspan=2)
        ttk.Entry(filtros, textvariable=self.limiar_inferior, width=7).grid(row=5, column=0)
        ttk.Entry(filtros, textvariable=self.limiar_superior, width=7).grid(row=5, column=1)
        ttk.Label(morfologia, text="Elemento quadrado:").grid(row=0, column=0)
        ttk.Combobox(morfologia, textvariable=self.tamanho_estrutura, values=(3, 5), state="readonly", width=5).grid(row=0, column=1)
        for linha, nome in enumerate(("Erosão", "Dilatação", "Abertura", "Fechamento"), start=1):
            self.botao_operacao(morfologia, nome, linha)
        self.botao_operacao(analise, "Histograma", 0)
        self.botao_operacao(analise, "Componentes e medidas", 1)
        ttk.Label(analise, text="Conectividade: 8\nÁrea em pixels²\nPerímetro e diâmetro em pixels").grid(row=2, column=0, columnspan=2, pady=5)
        ttk.Label(imagens, text="As operações usam o resultado atual. Restaure o original para começar novamente; branco = objeto.").grid(row=1, column=0, columnspan=4, sticky="w", pady=(6, 0))

        camera = ttk.LabelFrame(video, text="Câmera e processamento", padding=6)
        rastreamento = ttk.LabelFrame(video, text="Rastreamento CamShift", padding=6)
        deteccao = ttk.LabelFrame(video, text="Objeto específico + música", padding=6)
        for coluna, secao in enumerate((camera, rastreamento, deteccao)):
            secao.grid(row=0, column=coluna, sticky="nsew", padx=4)
            video.columnconfigure(coluna, weight=1)
        ttk.Label(camera, text="Índice da câmera:").grid(row=0, column=0)
        ttk.Entry(camera, textvariable=self.indice_camera, width=5).grid(row=0, column=1)
        ttk.Label(camera, text="Processamento:").grid(row=1, column=0)
        ttk.Combobox(camera, textvariable=self.modo_video, values=MODOS_VIDEO, state="readonly", width=15).grid(row=1, column=1)
        ttk.Label(camera, text="Filtro (lado):").grid(row=2, column=0)
        ttk.Combobox(camera, textvariable=self.tamanho_filtro, values=(3, 5, 7), state="readonly", width=5).grid(row=2, column=1)
        ttk.Label(camera, text="Canny inf./sup.:").grid(row=3, column=0)
        limiares_video = ttk.Frame(camera)
        limiares_video.grid(row=3, column=1)
        ttk.Entry(limiares_video, textvariable=self.limiar_inferior, width=5).pack(side="left", padx=2)
        ttk.Entry(limiares_video, textvariable=self.limiar_superior, width=5).pack(side="left", padx=2)
        ttk.Label(camera, text="Elemento (lado):").grid(row=4, column=0)
        ttk.Combobox(camera, textvariable=self.tamanho_estrutura, values=(3, 5), state="readonly", width=5).grid(row=4, column=1)
        ttk.Label(camera, text="Abra/feche a câmera na barra superior.").grid(row=5, column=0, columnspan=2)
        ttk.Label(camera, textvariable=self.info_camera, wraplength=270).grid(row=6, column=0, columnspan=2, pady=3)
        ttk.Button(rastreamento, text="Selecionar objeto", command=self.selecionar_objeto).pack(fill="x", pady=3)
        ttk.Button(rastreamento, text="Parar rastreamento", command=self.parar_rastreamento).pack(fill="x", pady=3)
        ttk.Label(rastreamento, text="Arraste na imagem da esquerda.\nA câmera pausa durante a seleção.\nSe o objeto sair, aguarda seu retorno.").pack(pady=8)
        ttk.Button(deteccao, text="Carregar referência", command=self.carregar_referencia).grid(row=0, column=0, sticky="ew")
        ttk.Button(deteccao, text="Selecionar música", command=self.selecionar_musica).grid(row=0, column=1, sticky="ew")
        ttk.Label(deteccao, textvariable=self.info_referencia, wraplength=350).grid(row=1, column=0, columnspan=2)
        ttk.Label(deteccao, textvariable=self.info_musica, wraplength=350).grid(row=2, column=0, columnspan=2)
        ttk.Label(deteccao, text="Limiar da referência:").grid(row=3, column=0)
        ttk.Entry(deteccao, textvariable=self.limiar_similaridade, width=8).grid(row=3, column=1)
        ttk.Button(deteccao, text="Ativar identificação", command=self.ativar_identificacao).grid(row=4, column=0, pady=3)
        ttk.Button(deteccao, text="Parar identificação/música", command=self.parar_identificacao).grid(row=4, column=1, pady=3)
        ttk.Button(deteccao, text="Testar música", command=self.testar_musica).grid(row=5, column=0, columnspan=2, sticky="ew", pady=3)
        ttk.Label(deteccao, textvariable=self.info_identificacao, wraplength=350).grid(row=6, column=0, columnspan=2, pady=3)
        ttk.Label(self.raiz, textvariable=self.status, relief="sunken", anchor="w", padding=6, wraplength=850).grid(row=3, column=0, sticky="ew")

    def botao_operacao(self, secao, nome, linha):
        botao = ttk.Button(secao, text=nome, command=lambda: self.aplicar(nome))
        botao.grid(row=linha, column=0, columnspan=2, sticky="ew", pady=2)
        self.botoes_imagem.append(botao)

    def mostrar_imagem(self, canvas, imagem):
        largura = max(canvas.winfo_width(), 2)
        altura = max(canvas.winfo_height(), 2)
        altura_imagem, largura_imagem = imagem.shape[:2]
        escala = min(largura / largura_imagem, altura / altura_imagem)
        nova_largura = max(1, int(largura_imagem * escala))
        nova_altura = max(1, int(altura_imagem * escala))
        # A inversão BGR -> RGB aqui só adapta a visualização do Pillow.
        rgb = imagem if imagem.ndim == 2 else imagem[:, :, ::-1]
        exibida = Image.fromarray(rgb).resize((nova_largura, nova_altura), Image.Resampling.BILINEAR)
        canvas.foto = ImageTk.PhotoImage(exibida)
        canvas.delete("all")
        deslocamento_x = (largura - nova_largura) // 2
        deslocamento_y = (altura - nova_altura) // 2
        canvas.create_image(deslocamento_x, deslocamento_y, anchor="nw", image=canvas.foto)
        return deslocamento_x, deslocamento_y, nova_largura / largura_imagem, nova_altura / altura_imagem

    def redesenhar_imagens(self):
        if self.captura is not None:
            if self.selecionando and self.frame_selecao is not None:
                self.transformacao_camera = self.mostrar_imagem(self.canvas_original, self.frame_selecao)
                self.inicio_selecao = None
            return
        if self.original is not None:
            self.mostrar_imagem(self.canvas_original, self.original)
            self.mostrar_imagem(self.canvas_resultado, self.visualizacao)
        else:
            for canvas, texto in ((self.canvas_original, "Carregue uma imagem ou clique em Abrir câmera."),
                                  (self.canvas_resultado, "O resultado do processamento aparecerá aqui.")):
                canvas.delete("all")
                canvas.create_text(canvas.winfo_width() // 2, canvas.winfo_height() // 2,
                                   text=texto, fill="white", width=max(canvas.winfo_width() - 20, 1))

    def carregar_imagem(self):
        if self.ocupado:
            return
        caminho = filedialog.askopenfilename(title="Carregar imagem", filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("Todos os arquivos", "*.*")])
        if not caminho:
            return
        try:
            imagem = ler_imagem(caminho)
        except (ValueError, OSError, cv2.error) as erro:
            messagebox.showerror("Não foi possível abrir", str(erro))
            return
        self.fechar_camera()
        self.abas.select(0)
        self.original = imagem
        self.atual = imagem.copy()
        self.visualizacao = self.atual
        self.redesenhar_imagens()
        self.status.set(f"Imagem carregada: {imagem.shape[1]} x {imagem.shape[0]} pixels. As operações usam a resolução original.")

    def trocar_aba(self, evento=None):
        if self.abas.index(self.abas.select()) == 0 and self.captura is not None:
            self.fechar_camera()
        self.habilitar_imagem()

    def exigir_imagem(self):
        if self.atual is None:
            messagebox.showwarning("Imagem necessária", "Carregue uma imagem antes de executar esta operação.")
            return False
        return True

    def restaurar(self):
        if self.exigir_imagem():
            self.atual = self.original.copy()
            self.visualizacao = self.atual
            self.redesenhar_imagens()
            self.status.set("Imagem original restaurada.")

    def salvar(self):
        if not self.exigir_imagem():
            return
        caminho = filedialog.asksaveasfilename(title="Salvar resultado exibido", defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp"), ("TIFF", "*.tiff")])
        if not caminho:
            return
        try:
            salvar_imagem(caminho, self.visualizacao)
            self.status.set(f"Resultado salvo em {caminho}")
        except (ValueError, OSError, cv2.error) as erro:
            messagebox.showerror("Erro ao salvar", str(erro))

    def parametros(self):
        try:
            tamanho = int(self.tamanho_filtro.get())
            estrutura = int(self.tamanho_estrutura.get())
            inferior = float(self.limiar_inferior.get())
            superior = float(self.limiar_superior.get())
        except ValueError:
            raise ValueError("Preencha tamanhos inteiros e limiares numéricos válidos.")
        pdi.validar_tamanho(tamanho)
        pdi.validar_tamanho(estrutura)
        if not 0 < inferior < superior <= 1500:
            raise ValueError("Canny: use 0 < inferior < superior <= 1500.")
        return tamanho, inferior, superior, estrutura

    def habilitar_imagem(self):
        estado = "normal" if not self.ocupado and self.captura is None else "disabled"
        for botao in self.botoes_imagem:
            botao.configure(state=estado)
        self.botao_carregar_imagem.configure(state="disabled" if self.ocupado else "normal")
        estado_camera = "disabled" if self.ocupado or self.captura is not None else "normal"
        self.botao_camera_topo.configure(state=estado_camera)

    def aplicar(self, nome):
        if self.ocupado or self.captura is not None or not self.exigir_imagem():
            return
        try:
            tamanho, inferior, superior, estrutura = self.parametros()
            if nome in ("Erosão", "Dilatação", "Abertura", "Fechamento", "Componentes e medidas"):
                pdi.exigir_binaria(self.atual)
        except ValueError as erro:
            messagebox.showwarning("Verifique a imagem/parâmetros", str(erro))
            return
        imagem = self.atual.copy()
        self.ocupado = True
        self.habilitar_imagem()
        self.status.set(f"Executando {nome} na resolução original… Imagens grandes podem levar alguns segundos.")

        def calcular():
            # A thread só calcula matrizes. Todos os widgets são atualizados na thread do Tkinter.
            try:
                dados = {"mensagem": f"{nome} concluído."}
                funcoes = {"Cinza": pdi.tons_de_cinza, "Negativo": pdi.negativo,
                           "Erosão": pdi.erosao, "Dilatação": pdi.dilatacao,
                           "Abertura": pdi.abertura, "Fechamento": pdi.fechamento}
                if nome == "Otsu":
                    dados["imagem"], limiar = pdi.otsu(imagem)
                    dados["mensagem"] = f"Otsu concluído. Limiar encontrado: {limiar}. Branco = intensidade maior que o limiar."
                elif nome == "Média":
                    dados["imagem"] = pdi.filtro_media(imagem, tamanho)
                elif nome == "Mediana":
                    dados["imagem"] = pdi.filtro_mediana(imagem, tamanho)
                elif nome == "Canny":
                    dados["imagem"] = pdi.canny(imagem, inferior, superior)
                elif nome == "Histograma":
                    dados["histograma"] = pdi.histograma(imagem)
                elif nome == "Componentes e medidas":
                    rotulos, objetos = pdi.componentes_conexos(imagem)
                    dados["medidas"] = pdi.medir_objetos(rotulos, objetos)
                    dados["cores"] = pdi.colorir_componentes(rotulos)
                    dados["mensagem"] = f"{len(objetos)} objeto(s). A imagem binária foi mantida para as próximas operações."
                elif nome in ("Cinza", "Negativo"):
                    dados["imagem"] = funcoes[nome](imagem)
                else:
                    dados["imagem"] = funcoes[nome](imagem, estrutura)
                self.fila_resultados.put((dados, None))
            except Exception as erro:
                self.fila_resultados.put((None, str(erro)))
        threading.Thread(target=calcular, daemon=True).start()

    def receber_resultado(self):
        try:
            dados, erro = self.fila_resultados.get_nowait()
        except queue.Empty:
            self.agendamento_resultado = self.raiz.after(80, self.receber_resultado)
            return
        self.ocupado = False
        self.habilitar_imagem()
        if erro:
            self.status.set("A operação não foi concluída.")
            messagebox.showerror("Erro na operação", erro)
        else:
            if "imagem" in dados:
                self.atual = dados["imagem"]
                self.visualizacao = self.atual
            if "cores" in dados:
                self.visualizacao = dados["cores"]
                self.mostrar_medidas(dados["medidas"])
            if "histograma" in dados:
                self.mostrar_histograma(dados["histograma"])
            self.redesenhar_imagens()
            self.status.set(dados["mensagem"])
        self.agendamento_resultado = self.raiz.after(80, self.receber_resultado)

    def mostrar_histograma(self, contagem):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        janela = tk.Toplevel(self.raiz)
        janela.title("Histograma calculado manualmente — luminância")
        figura = Figure(figsize=(7, 4), tight_layout=True)
        eixo = figura.add_subplot(111)
        eixo.bar(range(256), contagem, width=1, color="gray")
        eixo.set(xlabel="Intensidade", ylabel="Quantidade de pixels", xlim=(0, 255))
        grafico = FigureCanvasTkAgg(figura, master=janela)
        grafico.draw()
        grafico.get_tk_widget().pack(fill="both", expand=True)
        janela.grafico = grafico

    def mostrar_medidas(self, medidas):
        janela = tk.Toplevel(self.raiz)
        janela.title(f"Componentes conexos — {len(medidas)} objeto(s)")
        ttk.Label(janela, text="Branco = objeto | conectividade 8 | diâmetro entre centros dos pixels", padding=8).pack()
        quadro = ttk.Frame(janela)
        quadro.pack(fill="both", expand=True, padx=8, pady=8)
        tabela = ttk.Treeview(quadro, columns=("objeto", "area", "perimetro", "diametro"), show="headings", height=12)
        for coluna, titulo in (("objeto", "Objeto"), ("area", "Área (px²)"), ("perimetro", "Perímetro (px)"), ("diametro", "Diâmetro (px)")):
            tabela.heading(coluna, text=titulo)
            tabela.column(coluna, width=140, anchor="center")
        barra = ttk.Scrollbar(quadro, orient="vertical", command=tabela.yview)
        tabela.configure(yscrollcommand=barra.set)
        tabela.pack(side="left", fill="both", expand=True)
        barra.pack(side="right", fill="y")
        for medida in medidas:
            tabela.insert("", "end", values=(medida["objeto"], medida["area"], medida["perimetro"], f"{medida['diametro']:.3f}"))

    def abrir_camera(self):
        if self.captura is not None or self.ocupado:
            return
        try:
            indice = int(self.indice_camera.get())
            if not 0 <= indice <= 100:
                raise ValueError("O índice da câmera deve ser um inteiro entre 0 e 100.")
            self.parametros()
        except ValueError as erro:
            messagebox.showwarning("Parâmetros inválidos", str(erro))
            return
        try:
            captura = cv2.VideoCapture(indice)
        except cv2.error as erro:
            messagebox.showerror("Câmera indisponível", f"Não foi possível acessar a câmera: {erro}")
            return
        if not captura.isOpened():
            captura.release()
            messagebox.showerror("Câmera indisponível", "Não foi possível abrir a câmera. Verifique a conexão, permissões e o índice (0, 1…).")
            return
        captura.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.captura = captura
        self.abas.select(1)
        self.habilitar_imagem()
        self.status.set("Câmera aberta. Escolha o processamento ou selecione um objeto para rastrear.")
        self.atualizar_camera()
        if self.frame_atual is not None and self.referencia_ativa and self.caminho_musica is not None:
            self.ativar_identificacao("referencia")

    def atualizar_camera(self):
        self.agendamento_camera = None
        if self.captura is None:
            return
        if self.selecionando:
            self.agendamento_camera = self.raiz.after(30, self.atualizar_camera)
            return
        try:
            sucesso, frame = self.captura.read()
        except cv2.error:
            sucesso, frame = False, None
        if not sucesso or frame is None:
            self.fechar_camera()
            messagebox.showerror("Erro na câmera", "A câmera parou de fornecer frames. Tente abri-la novamente.")
            return
        self.frame_atual = frame
        try:
            tamanho, inferior, superior, estrutura = self.parametros()
            resultado = processar_video(frame, self.modo_video.get(), tamanho, inferior, superior, estrutura)
        except (ValueError, cv2.error) as erro:
            resultado = frame.copy()
            self.status.set(f"Vídeo exibido sem filtro: {erro}")
        # As marcações precisam de três canais, mesmo quando o filtro é binário.
        marcado = cv2.cvtColor(resultado, cv2.COLOR_GRAY2BGR) if resultado.ndim == 2 else resultado.copy()
        avisos = []
        if self.hist_rastreamento is not None:
            try:
                if self.rastreamento_perdido:
                    regiao = reencontrar_objeto(frame, self.referencia_rastreamento, self.hist_rastreamento)
                    if regiao is not None:
                        self.janela_rastreamento = regiao
                        self.rastreamento_perdido = False
                if not self.rastreamento_perdido:
                    caixa, self.janela_rastreamento = rastrear_frame(frame, self.hist_rastreamento, self.janela_rastreamento, self.referencia_rastreamento)
                    if caixa is None:
                        self.rastreamento_perdido = True
                    else:
                        pontos = cv2.boxPoints(caixa).astype(np.int32)
                        cv2.polylines(marcado, [pontos], True, (0, 255, 0), 2)
                        avisos.append("CamShift ativo (verde).")
                if self.rastreamento_perdido:
                    avisos.append("Objeto ausente. Aguardando seu retorno…")
                    cv2.putText(marcado, "Aguardando retorno do objeto", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            except cv2.error as erro:
                self.rastreamento_perdido = True
                avisos.append("Rastreamento aguardando nova posição.")
                self.status.set(f"Não foi possível localizar o objeto neste frame: {erro}")
        if self.identificacao_ativa or self.referencia_ativa:
            try:
                if self.fonte_identificacao == "selecao":
                    # O mesmo contorno verde visto pelo usuário alimenta o disparo de áudio.
                    encontrado = self.hist_rastreamento is not None and not self.rastreamento_perdido
                    avisos.append("Música vinculada ao objeto selecionado (verde).")
                    if encontrado:
                        cv2.putText(marcado, "Objeto selecionado: audio ativo", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    similaridade, (x, y, w, h) = localizar_referencia(frame, self.referencia)
                    limiar = self.ler_similaridade()
                    encontrado = similaridade >= limiar
                    avisos.append(f"Similaridade: {similaridade:.3f} / {limiar:.2f}")
                    if encontrado:
                        cv2.rectangle(marcado, (x, y), (x + w, y + h), (0, 255, 255), 2)
                        cv2.putText(marcado, "Objeto identificado", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                    if not self.identificacao_ativa:
                        if encontrado:
                            self.info_identificacao.set("Referência reconhecida (amarelo). Selecione uma música para ativar o áudio.")
                        else:
                            self.info_identificacao.set("Buscando a referência. Mostre o objeto com detalhes e orientação semelhantes.")
                if self.identificacao_ativa:
                    self.atualizar_audio(encontrado)
            except (ValueError, cv2.error, pygame.error, OSError) as erro:
                self.parar_identificacao()
                self.status.set(f"Identificação pausada: {erro}")
                self.info_identificacao.set(f"Identificação/áudio pausado: {erro}")
        self.info_camera.set(" | ".join(avisos) or f"Vídeo: {self.modo_video.get()}")
        self.transformacao_camera = self.mostrar_imagem(self.canvas_original, frame)
        self.mostrar_imagem(self.canvas_resultado, marcado)
        self.agendamento_camera = self.raiz.after(30, self.atualizar_camera)

    def fechar_camera(self):
        if self.agendamento_camera is not None:
            self.raiz.after_cancel(self.agendamento_camera)
            self.agendamento_camera = None
        if self.captura is not None:
            self.captura.release()
            self.captura = None
        self.frame_atual = None
        self.selecionando = False
        self.inicio_selecao = None
        self.frame_selecao = None
        self.canvas_original.configure(cursor="")
        self.parar_rastreamento()
        self.parar_identificacao()
        self.habilitar_imagem()
        self.canvas_original.delete("all")
        self.canvas_resultado.delete("all")
        self.redesenhar_imagens()
        self.info_camera.set("Câmera fechada.")
        self.status.set("Câmera fechada. As imagens estáticas foram preservadas.")
        if self.atual is not None:
            self.abas.select(0)

    def selecionar_objeto(self):
        if self.frame_atual is None:
            messagebox.showwarning("Câmera necessária", "Abra a câmera antes de selecionar o objeto.")
            return
        self.frame_selecao = self.frame_atual.copy()
        self.transformacao_camera = self.mostrar_imagem(self.canvas_original, self.frame_selecao)
        self.selecionando = True
        self.inicio_selecao = None
        self.canvas_original.configure(cursor="crosshair")
        self.status.set("Arraste um retângulo sobre o objeto na imagem da esquerda. A câmera está pausada.")

    def coordenadas_frame(self, x, y):
        deslocamento_x, deslocamento_y, escala_x, escala_y = self.transformacao_camera
        altura, largura = self.frame_selecao.shape[:2]
        fx = int((x - deslocamento_x) / escala_x)
        fy = int((y - deslocamento_y) / escala_y)
        return min(max(fx, 0), largura), min(max(fy, 0), altura)

    def iniciar_selecao(self, evento):
        if self.selecionando:
            self.inicio_selecao = (evento.x, evento.y)

    def mover_selecao(self, evento):
        if self.selecionando and self.inicio_selecao is not None:
            self.canvas_original.delete("selecao")
            x, y = self.inicio_selecao
            self.canvas_original.create_rectangle(x, y, evento.x, evento.y, outline="yellow", width=2, tags="selecao")

    def terminar_selecao(self, evento):
        if not self.selecionando or self.inicio_selecao is None:
            return
        x1, y1 = self.coordenadas_frame(*self.inicio_selecao)
        x2, y2 = self.coordenadas_frame(evento.x, evento.y)
        x, y = min(x1, x2), min(y1, y2)
        regiao = (x, y, abs(x2 - x1), abs(y2 - y1))
        selecao_valida = False
        try:
            hist = preparar_rastreamento(self.frame_selecao, regiao)
            self.parar_identificacao()
            self.hist_rastreamento = hist
            self.janela_rastreamento = regiao
            self.referencia_rastreamento = self.frame_selecao[y:y + regiao[3], x:x + regiao[2]].copy()
            self.rastreamento_perdido = False
            self.fonte_identificacao = "selecao"
            selecao_valida = True
            self.status.set("Rastreamento iniciado. Selecione uma música para tocar quando este objeto aparecer.")
        except (ValueError, cv2.error) as erro:
            messagebox.showwarning("Seleção inválida", str(erro))
        self.selecionando = False
        self.inicio_selecao = None
        self.frame_selecao = None
        self.canvas_original.delete("selecao")
        self.canvas_original.configure(cursor="")
        if selecao_valida and self.caminho_musica is not None:
            self.ativar_identificacao("selecao")

    def parar_rastreamento(self):
        if self.fonte_identificacao == "selecao":
            self.parar_identificacao()
        self.hist_rastreamento = None
        self.janela_rastreamento = None
        self.referencia_rastreamento = None
        self.rastreamento_perdido = False
        if self.selecionando:
            self.selecionando = False
            self.inicio_selecao = None
            self.frame_selecao = None
            self.canvas_original.delete("selecao")
            self.canvas_original.configure(cursor="")

    def carregar_referencia(self):
        caminho = filedialog.askopenfilename(title="Imagem de referência: recorte do objeto", filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.webp"), ("Todos os arquivos", "*.*")])
        if not caminho:
            return
        try:
            referencia = preparar_referencia(ler_imagem(caminho))
        except (ValueError, OSError, cv2.error) as erro:
            messagebox.showerror("Referência inválida", str(erro))
            return
        self.parar_identificacao()
        self.referencia = referencia
        self.referencia_ativa = True
        self.fonte_identificacao = "referencia"
        self.info_referencia.set(f"Referência: {os.path.basename(caminho)} ({referencia.shape[1]} x {referencia.shape[0]})")
        self.status.set("Referência carregada. A busca amarela funciona sem música. Mostre o objeto na câmera.")
        self.info_identificacao.set("Abra a câmera para buscar a referência; selecione uma música para ativar o áudio.")
        if self.frame_atual is not None and self.caminho_musica is not None:
            self.ativar_identificacao("referencia")

    def selecionar_musica(self):
        caminho = filedialog.askopenfilename(title="Selecionar música", filetypes=[("Áudio", "*.mp3 *.wav *.ogg"), ("Todos os arquivos", "*.*")])
        if not caminho:
            return
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(caminho)
        except (pygame.error, OSError) as erro:
            messagebox.showerror("Áudio indisponível", f"Não foi possível abrir a música/dispositivo de áudio: {erro}")
            return
        referencia_ativa = self.referencia_ativa
        self.parar_identificacao()
        self.caminho_musica = caminho
        self.info_musica.set(f"Música: {os.path.basename(caminho)}")
        if self.frame_atual is not None and (self.hist_rastreamento is not None or self.referencia is not None):
            self.ativar_identificacao()
        else:
            self.referencia_ativa = referencia_ativa
            self.status.set("Música carregada. Abra a câmera e selecione um objeto ou carregue uma referência.")
            self.info_identificacao.set("Música pronta. Selecione o objeto que deve disparar o áudio.")

    def ler_similaridade(self):
        try:
            limiar = float(self.limiar_similaridade.get().replace(",", "."))
        except ValueError:
            raise ValueError("A similaridade deve ser um número entre 0 e 1.")
        if not 0 < limiar <= 1:
            raise ValueError("A similaridade deve ser maior que 0 e menor ou igual a 1.")
        return limiar

    def ativar_identificacao(self, fonte=None):
        if self.frame_atual is None:
            messagebox.showwarning("Câmera necessária", "Abra a câmera antes de ativar a identificação.")
            return
        if fonte is None:
            fonte = self.fonte_identificacao
            if fonte == "selecao" and self.hist_rastreamento is None:
                fonte = None
            if fonte == "referencia" and self.referencia is None:
                fonte = None
            if fonte is None:
                if self.referencia is not None:
                    fonte = "referencia"
                elif self.hist_rastreamento is not None:
                    fonte = "selecao"
        selecao_disponivel = fonte == "selecao" and self.hist_rastreamento is not None
        referencia_disponivel = fonte == "referencia" and self.referencia is not None
        if not selecao_disponivel and not referencia_disponivel:
            messagebox.showwarning("Objeto necessário", "Selecione um objeto na câmera ou carregue uma imagem de referência.")
            return
        if self.caminho_musica is None and not referencia_disponivel:
            messagebox.showwarning("Música necessária", "Selecione a música que será tocada.")
            return
        try:
            if fonte == "referencia":
                self.ler_similaridade()
                localizar_referencia(self.frame_atual, self.referencia)
            if self.caminho_musica is not None:
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                pygame.mixer.music.load(self.caminho_musica)
        except (ValueError, cv2.error, pygame.error, OSError) as erro:
            messagebox.showerror("Não foi possível ativar", str(erro))
            return
        self.objeto_presente = False
        self.frames_ausentes = 0
        self.ultimo_disparo = -float("inf")
        self.musica_pendente = False
        self.musica_pausada = False
        self.fonte_identificacao = fonte
        self.referencia_ativa = fonte == "referencia"
        self.identificacao_ativa = self.caminho_musica is not None
        alvo = "objeto selecionado (verde)" if fonte == "selecao" else "imagem de referência (amarelo)"
        if self.identificacao_ativa:
            self.status.set(f"Áudio ativado para {alvo}. A música toca quando o objeto aparece.")
            self.info_identificacao.set(f"Áudio ativo: {alvo}.")
        else:
            self.status.set("Busca da referência ativada. Selecione uma música para vincular o áudio ao objeto.")
            self.info_identificacao.set("Buscando a referência (amarelo), sem música.")

    def tocar_musica(self):
        if self.caminho_musica is None:
            raise ValueError("Selecione uma música antes de reproduzir.")
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(self.caminho_musica)
        pygame.mixer.music.set_volume(1.0)
        pygame.mixer.music.play()
        self.ultimo_disparo = time.monotonic()
        self.musica_pendente = False
        self.musica_pausada = False

    def testar_musica(self):
        if self.caminho_musica is None:
            messagebox.showwarning("Música necessária", "Selecione uma música antes de testar o áudio.")
            return
        try:
            # O teste manual precisa tocar mesmo sem um objeto na câmera.
            self.parar_identificacao()
            self.tocar_musica()
            self.status.set("Teste de música iniciado. Use Parar identificação/música para interromper.")
            self.info_identificacao.set("Teste de áudio em reprodução. Use Ativar identificação para voltar ao modo automático.")
        except (ValueError, pygame.error, OSError) as erro:
            messagebox.showerror("Erro no áudio", f"Não foi possível reproduzir a música: {erro}")

    def atualizar_audio(self, encontrado):
        self.objeto_presente, self.frames_ausentes, disparar = atualizar_presenca(
            encontrado, self.objeto_presente, self.frames_ausentes, limite_ausencias=1
        )
        # Interrompe o som no primeiro frame sem reconhecimento. A pausa guarda
        # a posição da música; o retorno do objeto não recarrega o arquivo.
        if not encontrado:
            self.musica_pendente = False
            if not self.musica_pausada and pygame.mixer.music.get_busy():
                pygame.mixer.music.pause()
                self.musica_pausada = True
                self.status.set("Objeto ausente: música pausada.")
            if self.musica_pausada:
                self.info_identificacao.set("Música pausada. Aguardando reconhecer o objeto novamente.")
            else:
                self.info_identificacao.set("Áudio ativo. Aguardando o objeto aparecer.")
            return
        if self.musica_pausada:
            pygame.mixer.music.unpause()
            self.musica_pausada = False
            self.musica_pendente = False
            self.status.set("Objeto reconhecido novamente: música retomada.")
            self.info_identificacao.set("Música em reprodução.")
            return
        if disparar:
            self.musica_pendente = True
        # O intervalo vale para iniciar uma nova reprodução, nunca para retomar
        # uma música pausada. Não reinicia o áudio a cada frame reconhecido.
        if self.musica_pendente and time.monotonic() - self.ultimo_disparo >= 2:
            if not pygame.mixer.music.get_busy():
                self.tocar_musica()
                self.status.set("Objeto detectado: música iniciada.")
        if pygame.mixer.music.get_busy():
            self.info_identificacao.set("Música em reprodução.")
        elif self.musica_pendente:
            self.info_identificacao.set("Objeto presente. Aguardando o intervalo entre disparos.")
        else:
            self.info_identificacao.set("Objeto presente. A música já foi disparada nesta entrada.")

    def parar_identificacao(self):
        self.identificacao_ativa = False
        self.referencia_ativa = False
        self.objeto_presente = False
        self.frames_ausentes = 0
        self.musica_pendente = False
        self.musica_pausada = False
        self.info_identificacao.set("Áudio automático parado. Clique em Ativar identificação para retomar.")
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()

    def encerrar(self):
        self.raiz.after_cancel(self.agendamento_resultado)
        self.fechar_camera()
        if pygame.mixer.get_init():
            pygame.mixer.quit()
        self.raiz.destroy()


def main():
    raiz = tk.Tk()
    Aplicacao(raiz)
    raiz.mainloop()


if __name__ == "__main__":
    main()
