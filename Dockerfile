# Verwende ein offizielles Python-Laufzeit-Bild als Basisbild
FROM python:3.12-slim


# Installiere notwendige Pakete und Midnight Commander (mc)
RUN apt-get update && apt-get install -y \
    mc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


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
