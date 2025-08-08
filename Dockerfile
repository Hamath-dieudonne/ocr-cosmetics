# Utiliser une image Python officielle comme base
FROM python:3.11-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Installer dépendances système nécessaires à EasyOCR/OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 100 --retries 5 -r requirements.txt

# Copier le reste des fichiers du projet
COPY . .

# Exposer le port utilisé par FastAPI
EXPOSE 8000

# Commande pour lancer l'application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT", "--proxy-headers"]
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
