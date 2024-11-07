# Verwende das NVIDIA CUDA-Image (z. B. für CUDA 12.3 und Python 3.12)
FROM nvidia/cuda:12.3.0-devel-ubuntu20.04

# Installiere Python 3.12 und notwendige Pakete
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-distutils \
    python3-pip \
    mc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Setze das Arbeitsverzeichnis im Container
WORKDIR /app

# Kopiere die Anforderungen (requirements) Datei in das Arbeitsverzeichnis
COPY requirements.txt requirements.txt

# Installiere die Python-Abhängigkeiten (einschließlich TensorFlow-GPU)
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt

# Kopiere den Quellcode in das Arbeitsverzeichnis
COPY . .

# Main.py ausführbar machen
RUN chmod +x /app/main.py

# Standardbefehl zum Starten der Applikation
CMD ["python3.12", "main.py", "--help"]
