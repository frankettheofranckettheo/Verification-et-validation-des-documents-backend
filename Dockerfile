# 1. Utiliser une image Python légère officielle
FROM python:3.10-slim

# 2. Empêcher Python d'écrire des fichiers .pyc et activer les logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Installer les dépendances système
# CORRECTION ICI : Remplacement de libgl1-mesa-glx par libgl1 et ajout de libglib2.0-0
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-fra \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Créer le dossier de l'application
WORKDIR /app

# 5. Copier les requirements et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copier tout le reste du code source
COPY . .

# 7. Exposer le port
EXPOSE 10000

# 8. Commande de démarrage
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]