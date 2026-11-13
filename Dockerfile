FROM node:22-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip chromium \
    libnss3 libatk-bridge2.0-0 libgtk-3-0 libgbm1 \
    libasound2 libxshmfence1 libx11-xcb1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY whatsapp-agent/package*.json /app/whatsapp-agent/
COPY whatsapp-agent/scripts /app/whatsapp-agent/scripts
RUN cd /app/whatsapp-agent && npm ci

COPY . /app

RUN pip3 install --break-system-packages --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip3 install --break-system-packages --no-cache-dir \
    streamlit numpy pillow opencv-python-headless ultralytics \
    && pip3 uninstall -y opencv-python || true

ENV PROJECT_ROOT=/app
ENV PYTHON_EXECUTABLE=python3
ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium
ENV WHATSAPP_SESSION_DIR=/data/.wwebjs_auth

CMD ["npm", "--prefix", "/app/whatsapp-agent", "run", "start"]
