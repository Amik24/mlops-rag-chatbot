import os
import sys
import boto3
from pathlib import Path
from dotenv import load_dotenv # Gardé pour le test local

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Ajouter le chemin 'src' pour les imports si nécessaire lors de l'exécution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# --- Configuration des chemins et ressources ---
# Le nom du bucket S3 (doit correspondre à votre ressource)
S3_BUCKET_NAME = "g1-data" 
# Le préfixe où le CI/CD a uploadé les fichiers de l'index FAISS
S3_INDEX_PREFIX = "artifacts/vector_index/" 
# Le chemin local où l'index sera téléchargé et chargé sur le runner Streamlit
INDEX_PATH = "data/processed/faiss_index" 

class RAGModel:
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None
        
    def download_index_from_s3(self):
        """
        Télécharge les fichiers index.faiss et index.pkl depuis S3 vers le répertoire local du runner.
        """
        # 1. Créer le dossier local s'il n'existe pas
        Path(INDEX_PATH).mkdir(parents=True, exist_ok=True)
        
        # 2. Initialiser Boto3 (Il lit les variables d'environnement définies dans streamlit_app.py)
        # Nécessite AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN, AWS_REGION
        s3 = boto3.client('s3', region_name=os.environ.get("AWS_REGION"))
        
        # Les fichiers FAISS sont index.faiss et index.pkl
        required_files = ["index.faiss", "index.pkl"]
        
        print(f"Téléchargement de l'index depuis s3://{S3_BUCKET_NAME}/{S3_INDEX_PREFIX}")

        for filename in required_files:
            s3_key = S3_INDEX_PREFIX + filename
            local_path = Path(INDEX_PATH) / filename
            try:
                # 3. Téléchargement
                s3.download_file(S3_BUCKET_NAME, s3_key, str(local_path))
                print(f"-> Téléchargé : {filename}")
            except Exception as e:
                # Lève une erreur si l'index n'est pas sur S3 (signifiant que la CI/CD n'a pas tourné)
                raise FileNotFoundError(f"Erreur S3 : Impossible de télécharger {s3_key}. Détail: {e}")

    def load_model(self):
        """
        Gère le téléchargement de l'index S3, le chargement local et la configuration de la chaîne RAG.
        """
        # --- ÉTAPE CRUCIALE ---
        # 1. Télécharger l'index depuis S3 au démarrage de l'application Streamlit
        self.download_index_from_s3()

        # 2. Load Embeddings (Doit correspondre à celui utilisé dans build_embeddings.py)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # 3. Load Vector DB à partir du chemin local (où il vient d'être téléchargé)
        self.vector_store = FAISS.load_local(
            INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True # Requis pour charger des index provenant de sources externes
        )

        # 4. Initialize LLM (Lit GROQ_API_KEY depuis os.environ)
        llm = ChatGroq(
            temperature=0, 
            model_name="mixtral-8x7b-32768", 
            api_key=os.environ.get("GROQ_API_KEY") 
        )   

        # 5. Create Retrieval Chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True
        )
        print("✅ RAG Model Loaded Successfully.")

    def predict(self, query):
        """Exécute une prédiction et retourne la réponse et les sources."""
        if not self.qa_chain:
            # Si la chaîne n'a pas pu être chargée au démarrage, on lève une erreur
            raise Exception("RAG chain not initialized. Check S3 and API keys.")
        
        response = self.qa_chain.invoke({"query": query})
        
        answer = response['result']
        sources = [doc.metadata.get('source_file', 'Unknown') for doc in response['source_documents']]
        unique_sources = list(set(sources))
        
        return answer, unique_sources

# Test run local (Nécessite .env avec GROQ_API_KEY et l'index FAISS localement)
if __name__ == "__main__":
    load_dotenv() 
    # NOTE: Pour tester ce script localement, vous devez simuler la présence de l'index S3.
    # Vous pouvez sauter le download_index_from_s3() si vous avez créé l'index localement.
    
    rag = RAGModel()
    rag.load_model()
    ans, src = rag.predict("What is the Vanishing Gradient Problem?")
    print(f"Answer: {ans}")
    print(f"Sources: {src}")
