import os
import sys
import boto3
from pathlib import Path
from dotenv import load_dotenv 

# Imports des dépendances RAG
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- Configuration des chemins et ressources ---
S3_BUCKET_NAME = "g1-data"
S3_INDEX_PREFIX = "artifacts/vector_index/" 
INDEX_PATH = "data/processed/faiss_index" 

class RAGModel:
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None
        
    def download_index_from_s3(self):
        """Télécharge les fichiers de l'index FAISS depuis S3 vers le répertoire local."""
        
        # Le client Boto3 utilise os.environ pour lire les clés AWS/Token
        try:
            s3 = boto3.client('s3', region_name=os.environ.get("AWS_REGION"))
        except Exception as e:
            raise Exception(f"Erreur d'initialisation Boto3 (Clés/Région) : {e}")
            
        # 1. Créer le dossier local s'il n'existe pas
        Path(INDEX_PATH).mkdir(parents=True, exist_ok=True)
        
        required_files = ["index.faiss", "index.pkl"]
        print(f"Téléchargement de l'index depuis s3://{S3_BUCKET_NAME}/{S3_INDEX_PREFIX}")

        for filename in required_files:
            s3_key = S3_INDEX_PREFIX + filename
            local_path = Path(INDEX_PATH) / filename
            try:
                # 2. Téléchargement du fichier
                s3.download_file(S3_BUCKET_NAME, s3_key, str(local_path))
                print(f"-> Téléchargé : {filename}")
            except Exception as e:
                # Échoue si l'artefact n'est pas sur S3
                raise FileNotFoundError(f"Fichier S3 manquant : {s3_key}. L'erreur indique : {e}")

    def load_model(self):
        """
        Gère le téléchargement de l'index S3, le chargement local et la configuration de la chaîne RAG.
        """
        # 1. Télécharger l'index depuis S3 au démarrage
        self.download_index_from_s3()

        # 2. Load Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # 3. Load Vector DB à partir du chemin local
        self.vector_store = FAISS.load_local(
            INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )

        # 4. Initialize LLM (Lit GROQ_API_KEY depuis os.environ)
        llm = ChatGroq(
            temperature=0, 
            model_name="mixtral-8x7b-32768", 
            api_key=os.environ.get("GROQ_API_KEY") # Lit la clé Groq définie par Streamlit Secrets
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
        if not self.qa_chain:
            raise Exception("RAG chain not initialized.")
        
        response = self.qa_chain.invoke({"query": query})
        
        answer = response['result']
        sources = [doc.metadata.get('source_file', 'Unknown') for doc in response['source_documents']]
        unique_sources = list(set(sources))
        
        return answer, unique_sources

# Test run local
if __name__ == "__main__":
    load_dotenv() 
    # Pour le test local, vous devriez ajouter des variables AWS, GROQ_API_KEY dans un fichier .env
    # et vous assurer que l'index FAISS existe localement pour le test ou simuler un téléchargement.
    
    rag = RAGModel()
    rag.load_model()
    ans, src = rag.predict("What is the Vanishing Gradient Problem?")
    print(f"Answer: {ans}")
    print(f"Sources: {src}")
