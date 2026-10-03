# O Python e as bibliotecas ficam no contêiner; os arquivos do usuário ficam no host.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYGAME_HIDE_SUPPORT_PROMPT=1 \
    MPLCONFIGDIR=/tmp/matplotlib

# Tkinter precisa das bibliotecas Tcl/Tk. OpenCV e pygame precisam de vídeo/áudio.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tk \
        fonts-dejavu-core \
        libgl1 \
        libglib2.0-0 \
        libasound2 \
        libpulse0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip check

COPY main.py processamento.py ./
RUN python -c "import tkinter, cv2, numpy, PIL, matplotlib, pygame, main, processamento; print('Importacoes OK')" \
    && rm -rf /tmp/matplotlib \
    && groupadd --gid 1000 pdi \
    && useradd --uid 1000 --gid pdi --create-home pdi \
    && mkdir /dados

# O script Linux troca UID/GID pelos do usuário local para preservar permissões.
USER 1000:1000
WORKDIR /dados
CMD ["python", "/app/main.py"]
