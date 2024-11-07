# Verwende ein offizielles Python-Laufzeit-Bild als Basisbild
FROM python:3.12-slim


# Installiere notwendige Pakete und Midnight Commander (mc)
RUN apt-get update && apt-get install -y \
    mc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Installiere CUDA 12.3
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin | tee /etc/apt/preferences.d/cuda-repository-pin-600
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-12.3.0_520.61.05-1_amd64.deb -o cuda-repo.deb
RUN dpkg -i cuda-repo.deb
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN apt-get update
RUN apt-get install -y cuda-toolkit-12-3

# Setze das Arbeitsverzeichnis im Container
WORKDIR /app

# Kopiere die Anforderungen (requirements) Datei und den Quellcode in das Arbeitsverzeichnis
COPY requirements.txt requirements.txt
#COPY . .

# Installiere die Abhängigkeiten
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# main.py executable
RUN chmod +x /app/main.py

# Standardbefehl zum Starten der Applikation (kann zur Laufzeit überschrieben werden)
CMD ["python", "main.py", "--help"]
