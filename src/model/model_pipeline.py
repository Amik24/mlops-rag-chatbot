import os
import boto3
import shutil
import sys
from dotenv import load_dotenv

# --- IMPORTS LANGCHAIN STANDARD ---
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# --- CORRECTION ICI : ON UTILISE LES CHEMINS OFFICIELS ---
# (Pas de 'langchain_classic', ça n'existe pas !)
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# Configuration S3
# Note : Assure-toi que ces chemins correspondent bien à ce que tu as dans ton S3
S3_BUCKET_NAME = "g1-data"
S3_ARTIFACT_PATH = "artifacts/vector_index/faiss_index"
LOCAL_INDEX_PATH = "/tmp/faiss_index_g1"

class RAGModel:
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None

    def _download_index_from_s3(self):
        """Télécharge l'index FAISS depuis S3."""
        print(f"🔄 Téléchargement de l'index depuis S3 ({S3_BUCKET_NAME})...")

        # Nettoyage du dossier local s'il existe déjà pour éviter les conflits
        # Nettoyage du dossier temporaire
        if os.path.exists(LOCAL_INDEX_PATH):
            shutil.rmtree(LOCAL_INDEX_PATH)
        os.makedirs(LOCAL_INDEX_PATH)

        try:
            # Boto3 va utiliser les variables d'env chargées par streamlit_app.py
            s3 = boto3.client('s3')
            files = ["index.faiss", "index.pkl"]
            # CORRECTION : On force la région ici avec os.getenv
            region = os.getenv("AWS_REGION", "eu-west-3") # Par défaut eu-west-3 si non trouvé
            s3 = boto3.client('s3', region_name=region)

            # Téléchargement des 2 fichiers vitaux
            files = ["index.faiss", "index.pkl"]
            for file in files:
                source = f"{S3_ARTIFACT_PATH}/{file}"
                destination = f"{LOCAL_INDEX_PATH}/{file}"
                print(f"   📥 Téléchargement de {file}...")
                s3.download_file(S3_BUCKET_NAME, source, destination)
                # Construction du chemin S3 précis
                s3_key = f"{S3_ARTIFACT_PATH}/{file}"
                local_dest = f"{LOCAL_INDEX_PATH}/{file}"
                
                print(f"⬇️ Downloading {s3_key}...")
                s3.download_file(S3_BUCKET_NAME, s3_key, local_dest)

            print("✅ Index FAISS téléchargé avec succès.")
            print("✅ Index FAISS téléchargé.")
        except Exception as e:
            # Message d'erreur détaillé pour t'aider si S3 bloque
            raise Exception(f"Erreur S3 critique : Impossible de télécharger l'index depuis {S3_BUCKET_NAME}/{S3_ARTIFACT_PATH}. \nErreur: {str(e)}")
            # On affiche la région utilisée dans l'erreur pour le débogage
            current_region = os.getenv('AWS_REGION')
            raise Exception(f"Erreur S3 (Region: {current_region}) : Impossible de télécharger l'index. Détails: {e}")

    def load_model(self):
        # 1. Télécharger les données
@@ -113,3 +118,4 @@
            sources = [doc.metadata.get('source', 'Doc inconnu') for doc in response['context']]

        return response['answer'], sources
