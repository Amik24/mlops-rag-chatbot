import os
import sys
# Ajouter le chemin src pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Charge les clés localement pour le test (non nécessaire en CI/CD ou Streamlit Cloud)
load_dotenv() 

# Le modèle s'attend à trouver l'index ici (après téléchargement par Streamlit ou création par CI/CD)
INDEX_PATH = "data/processed/faiss_index" 

class RAGModel:
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None

    # NOTE: Cette méthode est faite pour le test local. L'app Streamlit gère le chargement
    # en téléchargeant d'abord l'index depuis S3.
    def load_model(self):
        """Loads the FAISS index and sets up the QA chain."""
        if not os.path.exists(INDEX_PATH):
            # Le chemin d'index est le dossier FAISS
            raise FileNotFoundError(f"❌ Erreur: Index FAISS non trouvé à {INDEX_PATH}. Lancez d'abord le pipeline de données.")

        # 1. Load Embeddings (Doit correspondre à celui utilisé dans build_embeddings.py)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # 2. Load Vector DB
        self.vector_store = FAISS.load_local(
            INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True # Requis si l'index vient d'une source non locale
        )

        # 3. Initialize LLM (Lit GROQ_API_KEY depuis l'environnement)
        llm = ChatGroq(
            temperature=0, 
            model_name="mixtral-8x7b-32768", 
            api_key=os.getenv("GROQ_API_KEY") 
        )   

        # 4. Create Retrieval Chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True
        )
        print("✅ RAG Model Loaded Successfully.")

    def predict(self, query):
        if not self.qa_chain:
            self.load_model()
        
        response = self.qa_chain.invoke({"query": query})
        
        answer = response['result']
        sources = [doc.metadata.get('source_file', 'Unknown') for doc in response['source_documents']]
        unique_sources = list(set(sources))
        
        return answer, unique_sources

# Test run
if __name__ == "__main__":
    # Assurez-vous d'avoir un fichier .env avec GROQ_API_KEY pour ce test
    rag = RAGModel()
    rag.load_model()
    ans, src = rag.predict("What is the Vanishing Gradient Problem?")
    print(f"Answer: {ans}")
    print(f"Sources: {src}")
