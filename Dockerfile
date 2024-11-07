# Verwende ein offizielles NVIDIA TensorFlow-Bild mit GPU-Unterstützung und CUDA 12.3 als Basis
FROM tensorflow/tensorflow:2.17.0-gpu

# Setze das Arbeitsverzeichnis im Container
WORKDIR /app

# Installiere notwendige Pakete und Midnight Commander (mc)
RUN apt-get update && apt-get install -y \
    mc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Kopiere die requirements.txt und installiere die Python-Abhängigkeiten
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Kopiere den gesamten Quellcode ins Arbeitsverzeichnis
COPY . .

# Mache main.py ausführbar
RUN chmod +x /app/main.py

# Standardbefehl zum Starten der Applikation (kann zur Laufzeit überschrieben werden)
CMD ["python", "main.py", "--help"]
