# Verwende Python 3.12 als Basisbild
FROM python:3.12-slim

# NVIDIA CUDA und cuDNN Schlüssel einfügen
RUN apt-get update && apt-get install -y wget gnupg && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update

# Installiere CUDA 12.3 und cuDNN 8.9
RUN apt-get install -y --no-install-recommends \
    cuda-toolkit-12-3 \
    libcudnn8=8.9.0.*-1+cuda12.3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Setze Umgebungsvariablen für CUDA und cuDNN
ENV PATH=/usr/local/cuda-12.3/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH

# Setze das Arbeitsverzeichnis im Container
WORKDIR /app

# Kopiere und installiere die Python-Abhängigkeiten
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Kopiere den gesamten Quellcode ins Arbeitsverzeichnis
COPY . .

# Mache main.py ausführbar
RUN chmod +x /app/main.py

# Standardbefehl zum Starten der Applikation (kann zur Laufzeit überschrieben werden)
CMD ["python", "main.py", "--help"]
