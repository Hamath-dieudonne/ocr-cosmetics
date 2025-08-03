# Utiliser une image Python officielle comme base
FROM python:3.11-slim
# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 100 --retries 5 -r requirements.txt

# Copier le reste des fichiers du projet
COPY . .

# Définir les variables d'environnement (optionnel, si .env est utilisé)
# Si vous ne voulez pas utiliser .env, laissez le chemin Tesseract par défaut
ENV TESSERACT_CMD=/usr/bin/tesseract

# Installer Tesseract-OCR dans le conteneur
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Exposer le port utilisé par FastAPI
EXPOSE 8000

# Commande pour lancer l'application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT", "--proxy-headers"]
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]