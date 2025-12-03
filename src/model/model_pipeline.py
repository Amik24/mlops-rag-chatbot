import os
import sys
import boto3
from pathlib import Path
from dotenv import load_dotenv # Maintenu pour le test local

# --- Imports LCEL/Moderne ---
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
# --- Fin Imports LCEL/Moderne ---

# Ajoutez le chemin 'src' pour les imports (si nécessaire)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# --- Configuration des chemins S3/Locaux ---
S3_BUCKET_NAME = "g1-data"
S3_INDEX_PREFIX = "artifacts/vector_index/" 
INDEX_PATH = "data/processed/faiss_index" 
# La variable INDEX_PATH doit être utilisée pour le chemin local

class RAGModel:
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None
        # Mettre à jour la variable pour qu'elle soit en snake_case
        self.INDEX_PATH = INDEX_PATH 

    def download_index_from_s3(self):
        """Télécharge les fichiers index.faiss et index.pkl depuis S3."""
        Path(self.INDEX_PATH).mkdir(parents=True, exist_ok=True)
        
        # Initialiser Boto3 (Il lit les clés SSO/Région depuis os.environ)
        try:
            s3 = boto3.client('s3', region_name=os.environ.get("AWS_REGION"))
        except Exception as e:
            raise Exception(f"Erreur d'initialisation Boto3 (Clés/Région) : {e}")

        required_files = ["index.faiss", "index.pkl"]
        print(f"Téléchargement de l'index depuis s3://{S3_BUCKET_NAME}/{S3_INDEX_PREFIX}")

        for filename in required_files:
            s3_key = S3_INDEX_PREFIX + filename
            local_path = Path(self.INDEX_PATH) / filename
            try:
                s3.download_file(S3_BUCKET_NAME, s3_key, str(local_path))
                print(f"-> Téléchargé : {filename}")
            except Exception as e:
                raise FileNotFoundError(f"Fichier S3 manquant : {s3_key}. Détail: {e}")
        
    def load_model(self):
        """
        Gère le téléchargement de l'index S3, le chargement local et la configuration de la chaîne RAG.
        """
        # --- 1. Télécharger l'index avant de le charger ---
        if not (Path(self.INDEX_PATH) / "index.faiss").exists():
            print("Index local non trouvé. Téléchargement depuis S3...")
            self.download_index_from_s3()
        else:
            print("Index trouvé localement.")

        # 2. Charger les Embeddings et le Vector Store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        self.vector_store = FAISS.load_local(
            self.INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )

        # 3. Initialize LLM (Groq)
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY manquante dans l'environnement Streamlit/CI/CD.")

        llm = ChatGroq(
            temperature=0.3, 
            model_name="llama-3.1-8b-instant",
            api_key=api_key
        )

        # 4. Create Prompt Template
        prompt = ChatPromptTemplate.from_template("""
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. Keep the answer concise.

        <context>
        {context}
        </context>

        Question: {input}
        """)

        # 5. Create the Document Chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)

        # 6. Create the Retrieval Chain
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        self.qa_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        print("✅ Modèle RAG chargé avec succès (LCEL).")

    def predict(self, query):
        if not self.qa_chain:
            raise Exception("RAG chain not initialized.")
            
        # LCEL utilise "input" comme clé d'entrée
        response = self.qa_chain.invoke({"input": query})
        
        # LCEL utilise "answer" comme clé de sortie pour la réponse
        answer = response.get('answer', 'Error: Answer not found.')
        
        # LCEL utilise "context" pour les documents sources
        sources = [doc.metadata.get('source_file', 'Inconnu') for doc in response.get('context', [])]
        unique_sources = list(set(sources))
        
        return answer, unique_sources

# Test run
if __name__ == "__main__":
    load_dotenv()
    if not os.path.exists(INDEX_PATH):
        print("NOTE: Index non trouvé. Lancer le pipeline de vectorisation d'abord.")
        sys.exit(1)
        
    rag = RAGModel()
    rag.load_model()
    print("--- Test Question ---")
    ans, src = rag.predict("What is the Vanishing Gradient Problem?")
    print(f"🤖 Réponse : {ans}")
    print(f"📚 Sources : {src}")
