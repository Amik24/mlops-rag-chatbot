import os
import boto3
import shutil
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- CORRECTION DES IMPORTS (V1.0+) ---
# Les chaînes sont maintenant dans langchain_classic
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Configuration S3
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
        
        if os.path.exists(LOCAL_INDEX_PATH):
            shutil.rmtree(LOCAL_INDEX_PATH)
        os.makedirs(LOCAL_INDEX_PATH)

        try:
            s3 = boto3.client('s3')
            files = ["index.faiss", "index.pkl"]
            for file in files:
                s3.download_file(S3_BUCKET_NAME, f"{S3_ARTIFACT_PATH}/{file}", f"{LOCAL_INDEX_PATH}/{file}")
            print("✅ Index FAISS téléchargé.")
        except Exception as e:
            raise Exception(f"Erreur S3 : Impossible de télécharger l'index. Avez-vous lancé le workflow GitHub ? Détails: {e}")

    def load_model(self):
        # 1. Télécharger
        self._download_index_from_s3()

        print("🧠 Chargement des Embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # 2. Charger FAISS
        self.vector_store = FAISS.load_local(
            LOCAL_INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )

        # 3. LLM (Groq)
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY manquante dans les secrets !")

        llm = ChatGroq(
            temperature=0.3, 
            model_name="llama-3.1-8b-instant",
            api_key=api_key
        )

        # 4. Prompt
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant for MLOps students.
        Answer based ONLY on the following context:
        <context>
        {context}
        </context>
        
        Question: {input}
        Answer:
        """)

        # 5. Chaîne (Via langchain_classic)
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        self.qa_chain = create_retrieval_chain(retriever, question_answer_chain)

    def predict(self, query):
        if not self.qa_chain:
            self.load_model()
        
        response = self.qa_chain.invoke({"input": query})
        return response['answer'], [doc.metadata.get('source_file', 'Doc') for doc in response['context']]
