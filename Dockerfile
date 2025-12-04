# 1. On part d'une version légère de Python 3.10 (celle qui marche chez vous)
FROM python:3.10-slim

# 2. On évite que Python garde des fichiers de cache inutiles
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. On crée le dossier de travail dans le conteneur
WORKDIR /app

# 4. On installe les dépendances système nécessaires pour AWS et Build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# 5. On copie d'abord les requirements pour profiter du cache Docker
COPY requirements.txt .

# 6. Installation des librairies Python
RUN pip install --no-cache-dir -r requirements.txt

# 7. On copie TOUT le reste du code (app/, src/, etc.) dans le conteneur
COPY . .

# 8. On indique sur quel port Streamlit écoute
EXPOSE 8501

# 9. La commande de lancement (Healthcheck inclus pour éviter les timeouts)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
