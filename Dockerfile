# Stage 1: Verwende das Python Image, um Python zu kompilieren
FROM python:3.12-slim AS python-build

# Alle Python-Dateien und -Ordner in einen Ordner kopieren
RUN mkdir -p /python/usr/local && cp -r /usr/local/* /python/usr/local/

# Stage 2: Verwende aktuelles CUDA Image
FROM nvidia/cuda:12.6.2-devel-ubuntu24.04

# Setze das Arbeitsverzeichnis im Container
WORKDIR /app
# Python-Dateien aus dem python-build-Image in das CUDA-Image kopieren
COPY --from=python-build /python/usr/local /usr/local

# Installiere notwendige Pakete und Midnight Commander (mc)
RUN apt-get update --allow-unauthenticated && apt-get install -y \
    build-essential cmake git zlib1g-dev \
    mc \
    graphviz \
    libgraphviz-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# DSSP aus Git bauen und Symlink setzen
RUN git clone https://github.com/PDB-REDO/dssp.git /tmp/dssp && \
    cd /tmp/dssp && \
    mkdir build && cd build && \
    cmake .. && \
    make && \
    cp dssp /usr/local/bin/mkdssp && \
    ln -sf /usr/local/bin/mkdssp /usr/bin/dssp && \
    rm -rf /tmp/dssp


# Kopiere die Anforderungen (requirements) Datei und den Quellcode in das Arbeitsverzeichnis
COPY requirements.txt requirements.txt
#COPY . .

# Installiere die Abhängigkeiten
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir pydot

COPY . .
# main.py executable
RUN chmod +x /app/main.py

# Standardbefehl zum Starten der Applikation (kann zur Laufzeit überschrieben werden)
CMD ["python", "main.py", "--help"]