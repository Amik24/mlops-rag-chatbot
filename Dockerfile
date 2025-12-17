# 1. Image de base
FROM python:3.10-slim

# 2. Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Répertoire de travail
WORKDIR /app

# 4. Installation des outils système
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 5. Copie et installation des dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copie du code source
COPY . .

# 7. Port
EXPOSE 8501

# 8. CONFIGURATION STREAMLIT FORCÉE (Fix)
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
headless = true\n\
address = \"0.0.0.0\"\n\
port = 8501\n\
" > /root/.streamlit/config.toml'

# 9. Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 10. Lancement (Plus simple maintenant car la config est dans le fichier toml)
ENTRYPOINT ["streamlit", "run", "src/app/streamlit_app.py"]
