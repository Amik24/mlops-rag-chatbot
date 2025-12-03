import os
import boto3
import shutil
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Configuration S3
S3_BUCKET_NAME = "g1-data"
# Chemin exact où le workflow GitHub a uploadé le dossier (recursive)
S3_ARTIFACT_PATH = "artifacts/vector_index/faiss_index"

# Chemin local temporaire pour Streamlit
LOCAL_INDEX_PATH = "/tmp/faiss_index_g1"

class RAGModel:
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None

    def _download_index_from_s3(self):
        """Télécharge l'index FAISS (dossier) depuis S3."""
        print(f"🔄 Tentative de téléchargement de l'index depuis S3 ({S3_BUCKET_NAME})...")
        
        if os.path.exists(LOCAL_INDEX_PATH):
            shutil.rmtree(LOCAL_INDEX_PATH)
        os.makedirs(LOCAL_INDEX_PATH)

        try:
            s3 = boto3.client('s3')
            # FAISS nécessite index.faiss et index.pkl
            files_to_download = ["index.faiss", "index.pkl"]
            
            for file in files_to_download:
                s3_key = f"{S3_ARTIFACT_PATH}/{file}"
                local_dest = f"{LOCAL_INDEX_PATH}/{file}"
                print(f"⬇️ Downloading {file}...")
                s3.download_file(S3_BUCKET_NAME, s3_key, local_dest)
            
            print("✅ Index FAISS téléchargé avec succès.")
            return True
        except Exception as e:
            print(f"❌ Erreur téléchargement S3: {e}")
            raise e

    def load_model(self):
        # 1. Télécharger l'index frais depuis S3
        self._download_index_from_s3()

        print("🧠 Chargement des Embeddings et du Vector Store...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # 2. Charger l'index localement
        self.vector_store = FAISS.load_local(
            LOCAL_INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )

        # 3. Initialiser LLM (Groq)
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ GROQ_API_KEY manquante dans les variables d'environnement")

        llm = ChatGroq(
            temperature=0.3, 
            model_name="llama-3.1-8b-instant",
            api_key=api_key
        )

        # 4. Créer le Prompt
        prompt = ChatPromptTemplate.from_template("""
        You are an assistant for question-answering tasks based on course materials.
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Keep the answer concise.

        <context>
        {context}
        </context>

        Question: {input}
        Answer:
        """)

        # 5. Créer la Chaîne RAG
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        self.qa_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        print("✅ Modèle RAG opérationnel.")

    def predict(self, query):
        if not self.qa_chain:
            self.load_model()
        
        response = self.qa_chain.invoke({"input": query})
        return response['answer'], [doc.metadata.get('source_file', 'Doc') for doc in response['context']]

if __name__ == "__main__":
    # Test local rapide
    rag = RAGModel()
    rag.load_model()
    print(rag.predict("What is NLP?"))

