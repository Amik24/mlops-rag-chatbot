# ... (imports conservés)
from .clean_transform import process_documents 

# MODIFICATION : Le chemin de sortie doit être 'models/faiss_index'
CI_CD_OUTPUT_PATH = "models/faiss_index"

def build_vector_store():
    # ... (les étapes 1, 2, 3 sont conservées) ...
    
    # 4. Save to Disk (Sauvegarde dans le chemin attendu par la CI/CD)
    if not os.path.exists(os.path.dirname(CI_CD_OUTPUT_PATH)):
        os.makedirs(os.path.dirname(CI_CD_OUTPUT_PATH))
    
    # MODIFICATION : Utiliser le nouveau chemin
    vector_store.save_local(CI_CD_OUTPUT_PATH) 
    print(f"✅ Vector Database saved to {CI_CD_OUTPUT_PATH}")

if __name__ == "__main__":
    build_vector_store()
