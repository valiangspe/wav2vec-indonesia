docker run --rm -it -p 5002:5002 --gpus all \
  -v $(pwd):/models \
  -v $(pwd)/speakers.pth:/root/speakers.pth \
  --entrypoint /bin/bash \
  ghcr.io/coqui-ai/tts
