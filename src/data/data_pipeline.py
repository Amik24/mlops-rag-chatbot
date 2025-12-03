from .download_data import load_data
from .build_embeddings import build_vector_store

def run_data_pipeline():
    print("--- Starting Data Vectorization Pipeline ---")
    
    # 1. Télécharger les données depuis S3
    load_data() 
    
    # 2. Nettoyer, Vectoriser, et Sauvegarder l'Index localement
    build_vector_store()
    
    print("--- Data Pipeline Completed (Ready for S3 Upload) ---")

if __name__ == "__main__":
    run_data_pipeline()
