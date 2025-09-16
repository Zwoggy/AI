FROM python:3.12-slim AS python-build

RUN mkdir -p /python/usr/local && cp -r /usr/local/* /python/usr/local/

FROM nvidia/cuda:12.6.2-devel-ubuntu24.04

WORKDIR /app
COPY --from=python-build /python/usr/local /usr/local

RUN apt-get update --allow-unauthenticated && apt-get install -y \
    build-essential cmake git zlib1g-dev \ libboost-all-dev \
    mc \
    graphviz \
    libgraphviz-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# DSSP aus Git bauen und Symlink setzen
RUN git clone https://github.com/PDB-REDO/dssp.git && \
  cd dssp && \
  cmake -S . -B build -DBUILD_PYTHON_MODULE=OFF && \
  cmake --build build && \
  cmake --install build



COPY requirements.txt requirements.txt
#COPY . .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir pydot

COPY . .
# main.py executable
RUN chmod +x /app/main.py

# Standardbefehl zum Starten der Applikation (kann zur Laufzeit überschrieben werden)
CMD ["python", "main.py", "--help"]

