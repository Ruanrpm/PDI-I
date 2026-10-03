#!/usr/bin/env bash
# Executa a GUI no desktop Linux, compartilhando somente os recursos necessários.
set -euo pipefail

imagem="${PDI_IMAGEM:-pdi-i:latest}"
camera="${PDI_CAMERA:-/dev/video0}"
usar_camera=true
usar_audio=true
reconstruir=false

for opcao in "$@"; do
    case "$opcao" in
        --build) reconstruir=true ;;
        --sem-camera) usar_camera=false ;;
        --sem-audio) usar_audio=false ;;
        --help|-h)
            echo "Uso: bash executar-docker.sh [--build] [--sem-camera] [--sem-audio]"
            echo "PDI_CAMERA=/dev/video2 escolhe outra câmera (no app, use índice 0)."
            echo "PDI_IMAGEM=pdi-i:latest escolhe a imagem Docker."
            exit 0 ;;
        *) echo "Opção desconhecida: $opcao. Use --help." >&2; exit 1 ;;
    esac
done

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Execute este script no desktop Linux da faculdade." >&2
    exit 1
fi
for comando in docker xauth; do
    if ! command -v "$comando" >/dev/null 2>&1; then
        echo "Comando ausente: $comando. Consulte COMANDOS_FACULDADE.txt." >&2
        exit 1
    fi
done
if ! docker info >/dev/null 2>&1; then
    echo "Docker indisponível. Verifique o serviço e a permissão do usuário." >&2
    exit 1
fi
if [[ -z "${DISPLAY:-}" || ! -d /tmp/.X11-unix ]]; then
    echo "Abra o terminal na sessão gráfica Linux (X11 ou Wayland com XWayland)." >&2
    exit 1
fi

pasta_projeto="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if $reconstruir || ! docker image inspect "$imagem" >/dev/null 2>&1; then
    docker build --tag "$imagem" "$pasta_projeto"
fi

# Copia a autorização da sessão para um arquivo temporário, sem usar 'xhost +'.
autorizacao="$(mktemp /tmp/pdi-xauth.XXXXXX)"
trap 'rm -f -- "$autorizacao"' EXIT
if ! xauth nlist "$DISPLAY" | sed 's/^..../ffff/' | xauth -f "$autorizacao" nmerge -; then
    echo "Falha ao copiar a autorização X11 da sessão. Confira DISPLAY e XAUTHORITY." >&2
    exit 1
fi
if [[ ! -s "$autorizacao" ]]; then
    echo "Não foi encontrada autorização X11. Execute como usuário da sessão gráfica." >&2
    exit 1
fi

argumentos=(
    --rm --init --name pdi-i-app
    --user "$(id -u):$(id -g)"
    --security-opt no-new-privileges
    --cap-drop ALL
    --network none
    --env "DISPLAY=$DISPLAY"
    --env XAUTHORITY=/tmp/pdi.xauth
    --volume /tmp/.X11-unix:/tmp/.X11-unix:ro
    --volume "$autorizacao:/tmp/pdi.xauth:ro"
    --volume "$pasta_projeto:/dados:rw"
)

if $usar_camera && [[ -c "$camera" ]]; then
    argumentos+=(--device "$camera:/dev/video0" --group-add "$(stat -c '%g' "$camera")")
    echo "Câmera: $camera (índice 0 dentro do aplicativo)."
elif $usar_camera; then
    echo "Câmera não encontrada em $camera. As operações de imagem continuam disponíveis."
fi

# PulseAudio e o serviço pipewire-pulse usam este socket da sessão do usuário.
socket_audio="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/pulse/native"
if $usar_audio && [[ -S "$socket_audio" ]]; then
    argumentos+=(
        --env SDL_AUDIODRIVER=pulseaudio
        --env PULSE_SERVER=unix:/tmp/pdi-pulse
        --volume "$socket_audio:/tmp/pdi-pulse"
    )
    cookie_audio="${PULSE_COOKIE:-${XDG_CONFIG_HOME:-$HOME/.config}/pulse/cookie}"
    if [[ -f "$cookie_audio" ]]; then
        argumentos+=(--env PULSE_COOKIE=/tmp/pdi-pulse-cookie --volume "$cookie_audio:/tmp/pdi-pulse-cookie:ro")
    fi
    echo "Áudio: servidor PulseAudio/pipewire-pulse da sessão."
elif $usar_audio && [[ -d /dev/snd ]]; then
    argumentos+=(--env SDL_AUDIODRIVER=alsa --device /dev/snd)
    # Dispositivos ALSA podem pertencer a grupos diferentes; inclui todos eles.
    for dispositivo in /dev/snd/*; do
        if [[ -c "$dispositivo" ]]; then
            argumentos+=(--group-add "$(stat -c '%g' "$dispositivo")")
        fi
    done
    echo "Áudio: dispositivos ALSA. Se estiverem ocupados, use o servidor de áudio da sessão."
else
    echo "Sem dispositivo de áudio compartilhado. O programa informará se a reprodução falhar."
fi

echo "Arquivos disponíveis em /dados. Salve os resultados nessa pasta para mantê-los."
docker run "${argumentos[@]}" "$imagem"
