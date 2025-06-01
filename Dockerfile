FROM python:3.10-slim

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Répertoire de l'app
WORKDIR /app

# Copier tous les fichiers du projet
COPY . /app

# Installer les dépendances Python
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Exposer le port utilisé par FastAPI
EXPOSE 8080

# Commande de lancement de l'API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
