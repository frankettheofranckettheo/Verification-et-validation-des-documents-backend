# 1. Utiliser une image Python légère officielle
FROM python:3.10-slim

# 2. Empêcher Python d'écrire des fichiers .pyc et activer les logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Installer les dépendances système (C'est ici qu'on règle le souci Tesseract !)
# - tesseract-ocr : le moteur OCR
# - tesseract-ocr-fra : le pack de langue français (important pour vos documents)
# - libgl1-mesa-glx : indispensable pour OpenCV (utilisé par Ultralytics) sinon ça crashe
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-fra \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 4. Créer le dossier de l'application
WORKDIR /app

# 5. Copier les requirements et installer les dépendances Python
# On le fait avant de copier le code pour profiter du cache Docker (build plus rapide)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copier tout le reste du code source
COPY . .

# 7. Exposer le port (Render utilise souvent le 10000 par défaut)
EXPOSE 10000

# 8. Commande de démarrage
# On utilise la variable d'environnement $PORT fournie par Render, ou 10000 par défaut
CMD ["sh", "-c", "uvicorn main_regularized:app --host 0.0.0.0 --port ${PORT:-10000}"]