# PDI-I | Processamento Digital de Imagens

Aplicação acadêmica em Python para processamento de imagens, análise de objetos
e demonstrações de vídeo em tempo real. A interface Tkinter permite comparar
a imagem original com o resultado e ajustar os parâmetros dos algoritmos.

As operações de **imagens estáticas são implementadas manualmente**, com NumPy
e algoritmos clássicos. No vídeo, as funções do OpenCV mantêm a captura
responsiva. O projeto preserva uma estrutura simples, adequada ao estudo e à
apresentação em uma disciplina de Processamento Digital de Imagens.

## Funcionalidades

| Área | Recursos |
| --- | --- |
| Aquisição | Carregar arquivos, restaurar a imagem original e salvar resultados |
| Conversões | Luminância em tons de cinza, negativo e limiar automático de Otsu |
| Filtros | Média e mediana com máscaras ajustáveis; Canny com dois limiares |
| Morfologia | Erosão, dilatação, abertura e fechamento em imagens binárias |
| Análise | Histograma, componentes conexos coloridos, contagem, área, perímetro e diâmetro |
| Vídeo | Câmera ao vivo com conversões, filtros e morfologia |
| Rastreamento | Seleção de uma região, CamShift e busca do objeto após desaparecer |
| Identificação | Imagem de referência, comparação em diferentes tamanhos e limiar de similaridade |
| Áudio | Música vinculada ao objeto; pausa na ausência e retomada no reconhecimento |

## Execução com Docker no Linux

### Requisitos do computador

- Docker Engine local em um computador Linux com sessão gráfica.
- Usuário autorizado a executar `docker` sem `sudo`.
- `xauth` e uma sessão X11, ou Wayland com XWayland e `DISPLAY` definido.
- Câmera V4L2, como `/dev/video0`, para as demonstrações de vídeo.
- PulseAudio/pipewire-pulse ou dispositivos ALSA para reprodução de áudio.

Python, Tkinter e as bibliotecas do aplicativo são instalados **na imagem
Docker**. A janela é exibida no desktop Linux; não há servidor web nem acesso
pelo navegador. Os comandos essenciais para instalação, clonagem e execução
estão em [COMANDOS_FACULDADE.txt](COMANDOS_FACULDADE.txt).

### Início rápido

Dentro da pasta do projeto, no terminal da sessão gráfica:

```bash
docker build -t pdi-i:latest .
bash executar-docker.sh
```

O script reutiliza a imagem existente e constrói uma nova somente se ela não
estiver disponível ou se `--build` for informado. Após alterar o código:

```bash
bash executar-docker.sh --build
```

### Levar a aplicação pronta para a faculdade

No computador de preparação, com Docker em modo de contêineres Linux:

```bash
docker build --platform linux/amd64 -t pdi-i:latest .
docker save -o pdi-i-linux-amd64.tar pdi-i:latest
```

Leve o TAR, a pasta do projeto e suas imagens/músicas. Não é necessário levar
`.venv`. No Linux da faculdade, dentro da pasta copiada:

```bash
docker load -i pdi-i-linux-amd64.tar
bash executar-docker.sh
```

Com Docker e `xauth` já instalados, esse fluxo não precisa de internet.
A imagem exportada é para **x86_64/amd64**; outro tipo de processador exige uma
construção compatível. Os comandos `save` e `load` transportam a imagem e suas
dependências. [Exportação de imagens](https://docs.docker.com/reference/cli/docker/image/save/),
[importação de imagens](https://docs.docker.com/reference/cli/docker/image/load/).

### Arquivos e persistência

O script monta a pasta do projeto como `/dados`, usando o UID/GID do usuário
Linux. Coloque imagens, referências e músicas dentro dessa pasta antes de iniciar.
Sugestão de organização dos arquivos pessoais:

```text
imagens/       imagens de entrada e referências
musicas/       arquivos MP3, WAV ou OGG
resultados/    imagens processadas
```

No diálogo de arquivos da aplicação, use `/dados` e suas subpastas. Salve em
`/dados/resultados` para preservar o resultado depois de encerrar o contêiner.
Músicas e fotos não entram na imagem Docker e devem acompanhar o projeto.

### Câmera e som

O script compartilha a câmera padrão quando ela existe. Para escolher outra:

```bash
PDI_CAMERA=/dev/video2 bash executar-docker.sh
```

No aplicativo, use **índice 0**: o dispositivo escolhido é mapeado para
`/dev/video0` dentro do contêiner. Conecte a câmera antes de iniciar e feche
programas que estejam usando o mesmo dispositivo.

Para áudio, o script prioriza o socket PulseAudio/pipewire-pulse da sessão.
Quando ele não existe, compartilha `/dev/snd` e os grupos dos dispositivos ALSA.
Use **Testar música** para conferir a saída de som e, depois,
**Ativar identificação** para retornar ao controle automático pelo objeto.

Para trabalhar apenas com imagens:

```bash
bash executar-docker.sh --sem-camera --sem-audio
```

Este fluxo de dispositivos é destinado ao **Docker Engine nativo no Linux**.
O Docker Desktop executa uma VM e não oferece passagem direta automática de
dispositivos USB. [Documentação do Docker Desktop](https://docs.docker.com/desktop/troubleshoot-and-support/faqs/general/#can-i-pass-through-a-usb-device-to-a-container).

## Execução nativa

Também é possível executar fora do Docker. Recomenda-se Python 3.12 ou 3.13.

### Windows

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

### Ubuntu/Debian

```bash
sudo apt-get install -y python3-venv python3-tk
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
```

| Biblioteca | Finalidade |
| --- | --- |
| NumPy | Matrizes de pixels e operações numéricas básicas |
| OpenCV | Arquivos, captura, processamento de vídeo, CamShift e template matching |
| Tkinter | Interface gráfica; componente do Python com suporte Tcl/Tk no sistema |
| Pillow | Exibição das imagens na interface |
| Matplotlib | Desenho do histograma calculado pelo programa |
| Pygame | Carregamento, reprodução, pausa e retomada da música |

As dependências instaladas por `pip` estão em [requirements.txt](requirements.txt).

## Como utilizar

1. Clique em **Carregar imagem**. A original aparece à esquerda e o resultado,
   à direita. As operações seguintes usam o resultado atual.
2. Use **Restaurar original** para comparar operações independentes. Ajuste
   as máscaras dos filtros, os limiares do Canny e o elemento estruturante.
3. Para morfologia e componentes, aplique **Otsu** antes: branco representa
   o objeto. Se necessário, use negativo após Otsu para inverter o fundo.
4. Abra **Histograma** ou **Componentes e medidas** para consultar o gráfico
   e a tabela. Salvar após a análise de componentes grava a visualização colorida.
5. Clique em **Abrir câmera**, na barra superior, e escolha um modo de vídeo.
   **Selecionar objeto** permite arrastar uma região na imagem da esquerda.
6. Selecione uma música para vincular o som ao objeto de contorno **verde**.
   A música pausa na ausência e retoma quando o reconhecimento volta.
7. Como alternativa, use **Carregar referência** com um recorte do objeto.
   A busca **amarela** funciona mesmo sem música; ajuste o limiar observando
   a similaridade. Selecionar música habilita o áudio nesse modo.
8. Feche a câmera ou a janela ao terminar. Carregar uma foto durante o vídeo
   encerra a câmera e retorna aos controles de imagem.

## Algoritmos e critérios de medição

### Imagens estáticas

- **Cinza:** luminância `0,299R + 0,587G + 0,114B`, respeitando a ordem BGR,
  arredondamento e intervalo de 0 a 255.
- **Negativo:** subtração de cada intensidade de 255.
- **Histograma e Otsu:** contagem manual das intensidades; teste dos limiares
  pela variância entre classes para produzir uma imagem binária.
- **Média e mediana:** vizinhanças por pixel; soma/média ou ordenação e valor
  central. As bordas repetem os pixels mais próximos.
- **Canny:** suavização gaussiana, Sobel X/Y, magnitude/direção, supressão de
  não máximos, limiarização dupla e histerese por busca em largura.
- **Morfologia:** elemento quadrado de 3×3 ou 5×5; fora da imagem é fundo.
  Abertura é erosão seguida de dilatação; fechamento faz a ordem inversa.

Essas implementações estão em [processamento.py](processamento.py), que não
importa OpenCV. As operações prontas equivalentes são usadas apenas no vídeo.

### Componentes conexos

A rotulação percorre os pixels brancos sem rótulo e inicia uma **BFS com
conectividade 8** para cada novo objeto. Retorna a matriz de rótulos e a lista
de pixels dos componentes; a contagem corresponde ao número dessas listas.

| Medida | Cálculo | Unidade |
| --- | --- | --- |
| Área | Quantidade de pixels do componente | pixels² |
| Perímetro | Soma dos lados expostos ao fundo, usando os quatro vizinhos | pixels |
| Diâmetro | Maior distância euclidiana entre centros de pixels de borda | pixels |

O perímetro inclui buracos internos. Um pixel isolado tem diâmetro zero.
As medidas são discretas e não representam centímetros sem calibração.

### Rastreamento, referência e música

CamShift usa o histograma de matiz da região selecionada em HSV. Se o objeto
desaparecer, o modelo é mantido e uma busca por aparência/cor tenta encontrá-lo
novamente. Na identificação por arquivo, template matching compara nove
tamanhos da referência, entre 50% e 200%; fotos grandes são reduzidas a um
modelo com até 160 pixels no maior lado, preservando a proporção.

O áudio tem estado próprio: inicia uma vez na entrada, pausa no primeiro frame
sem reconhecimento e retoma do mesmo ponto no retorno. Não reinicia a cada
frame. O intervalo de dois segundos vale apenas para novos disparos de uma
música que já terminou. **Parar identificação/música** encerra esse modo.

## Estrutura do projeto

```text
PDI-I/
├── main.py                    # Interface, arquivos, vídeo, identificação e áudio
├── processamento.py           # Algoritmos manuais de imagens
├── requirements.txt           # Dependências Python
├── Dockerfile                 # Ambiente Linux com Python e bibliotecas
├── .dockerignore              # Arquivos permitidos no contexto da construção
├── .gitattributes             # Finais de linha Linux nos scripts
├── executar-docker.sh         # Tela, câmera, som e pasta compartilhada
├── COMANDOS_FACULDADE.txt      # Instalação, execução e transporte
└── README.md                  # Documentação do projeto
```

Pastas de imagens, músicas e resultados são opcionais. A aplicação usa dois
módulos Python e uma classe para guardar o estado da interface; os algoritmos
são funções independentes. Operações manuais usam uma thread e uma fila para
manter a janela responsiva.

## Diagnóstico

| Situação | Verificação |
| --- | --- |
| Docker não responde | Execute `docker info`; confira o serviço e a permissão do usuário |
| Janela não abre | Use o terminal da sessão gráfica, confira `DISPLAY` e `xauth`; evite executar com `sudo` |
| Câmera não abre | Confira `/dev/video*`, use `PDI_CAMERA` e mantenha índice 0 no app |
| Música não toca | Use **Testar música**, confira o volume e o servidor de áudio da sessão |
| Referência não reconhece | Use um recorte com contraste e pouco fundo; ajuste orientação, distância e limiar |
| Resultado desaparece ao encerrar | Salve dentro de `/dados`, que corresponde à pasta local do projeto |
| Contêiner já está aberto | Feche a janela anterior ou execute `docker stop pdi-i-app` |

## Limitações e validação

Os algoritmos manuais priorizam clareza e podem demorar em imagens grandes.
O diâmetro compara pares de pixels de borda e tem custo quadrático. CamShift
e template matching dependem de aparência, cor, orientação e iluminação;
não são reconhecimento genérico de objetos ou identificação biométrica.

Foi construída uma imagem Linux/amd64 com Python 3.12. A validação incluiu
importações e `pip check`, GUI e `mainloop` com Xvfb, filtros, histograma,
tabela de medidas, leitura/salvamento e os 11 modos de vídeo com frames
sintéticos. O MP3 foi decodificado, pausado e retomado com o driver de áudio
`dummy`; isso verifica o fluxo do programa, sem comprovar audição física.
O lançador passou em `bash -n` e ShellCheck, além de verificações de autorização
X11, UID/GID, argumentos de dispositivos, reutilização da imagem e suas opções.
A imagem final também passou na execução com UID 1001, sem rede nem capacidades
Linux adicionais. O TAR exportado foi importado novamente com sucesso.

Acesso físico à câmera, exibição no desktop e audição no Linux da faculdade
precisam ser conferidos nesse equipamento. Ausência de câmera não impede
processar imagens; ausência de áudio é informada ao tentar reproduzir.

Para verificar as dependências da imagem:

```bash
docker run --rm pdi-i:latest python -m pip check
docker run --rm pdi-i:latest python -c "import tkinter, cv2, numpy, PIL, matplotlib, pygame; print('Importacoes OK')"
```

## Referências técnicas

- [Instalação do Docker Engine](https://docs.docker.com/engine/install/)
- [Execução de contêineres, dispositivos e volumes](https://docs.docker.com/engine/containers/run/)
- [Autorização de sessão X11 com xauth](https://www.x.org/archive/X11R6.8.1/doc/xauth.1.html)
- [CamShift no OpenCV](https://docs.opencv.org/4.x/d7/d00/tutorial_meanshift.html)
- [Template matching no OpenCV](https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html)
