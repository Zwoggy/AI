# Stage 1: Verwende das Python Image, um Python zu kompilieren
FROM python:3.12-slim AS python-build

# Alle Python-Dateien und -Ordner in einen Ordner kopieren
RUN mkdir -p /python/usr/local && cp -r /usr/local/* /python/usr/local/

# Stage 2: Verwende aktuelles CUDA Image
FROM nvidia/cuda:12.6.2-devel-ubuntu24.04


# Python-Dateien aus dem python-build-Image in das CUDA-Image kopieren
COPY --from=python-build /python/usr/local /usr/local

# Installiere notwendige Pakete und Midnight Commander (mc)
RUN apt-get update && apt-get install -y \
    mkdssp \
    mc \
    graphviz \
    libgraphviz-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Setze das Arbeitsverzeichnis im Container
WORKDIR /app

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