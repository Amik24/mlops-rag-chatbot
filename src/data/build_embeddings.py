import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

try:
    from clean_transform import process_documents
except ImportError:
    from src.data.clean_transform import process_documents

# CHEMIN CRITIQUE POUR LA CI/CD
CI_CD_OUTPUT_PATH = "models/faiss_index"

def build_vector_store():
    print("--- Démarrage de la construction de l'index Vectoriel ---")

    # 1. Récupérer les chunks (via clean_transform.py)
    chunks = process_documents()
    if not chunks:
        print("❌ Stop : Aucun chunk de texte généré.")
        return

    # 2. Initialiser le modèle d'embeddings
    print("⏳ Chargement du modèle d'embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3. Créer le Vector Store FAISS
    print(f"⚡ Création de l'index FAISS avec {len(chunks)} chunks...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # 4. Sauvegarder sur le disque (pour l'upload S3)
    if not os.path.exists(os.path.dirname(CI_CD_OUTPUT_PATH)):
        os.makedirs(os.path.dirname(CI_CD_OUTPUT_PATH))
    
    vector_store.save_local(CI_CD_OUTPUT_PATH)
    print(f"✅ Index FAISS sauvegardé avec succès dans : {CI_CD_OUTPUT_PATH}")
    print("🚀 Prêt pour l'upload S3 par le workflow CI/CD.")

if __name__ == "__main__":
    build_vector_store()
