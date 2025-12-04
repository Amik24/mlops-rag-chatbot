# 1. Image de base (Python 3.10 comme sur Streamlit Cloud)
FROM python:3.10-slim

# 2. Variables d'environnement pour optimiser Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Répertoire de travail
WORKDIR /app

# 4. Installation des outils système (nécessaire pour construire certaines libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# 5. Copie et installation des dépendances (Cache Docker optimisé)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copie du code source
COPY . .

# 7. Port Streamlit par défaut
EXPOSE 8501

# 8. Vérification de santé (Healthcheck)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 9. Commande de lancement
ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
