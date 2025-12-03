import sys
import os

# Ajout du chemin racine pour permettre les imports entre modules si nécessaire
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from src.data.download_data import load_data
    from src.data.build_embeddings import build_vector_store
except ImportError:
    # Fallback pour exécution locale directe dans le dossier
    from download_data import load_data
    from build_embeddings import build_vector_store

def run_data_pipeline():
    print("==========================================")
    print("🚀 DÉMARRAGE DU PIPELINE DE VECTORISATION")
    print("==========================================")
    
    # Étape 1 : Téléchargement (S3 -> Local)
    print("\n[ÉTAPE 1/2] Téléchargement des données...")
    load_data() 
    
    # Étape 2 : Vectorisation (Local -> models/)
    print("\n[ÉTAPE 2/2] Construction de l'index...")
    build_vector_store()
    
    print("\n==========================================")
    print("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
    print("==========================================")

if __name__ == "__main__":
    run_data_pipeline()
