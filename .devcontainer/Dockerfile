FROM python:3.11-slim

# Installer les paquets nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    tesseract-ocr \
    locales \
    git \
    && rm -rf /var/lib/apt/lists/*

# Configurer la locale en_US.UTF-8
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Copier et installer requirements Python
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
