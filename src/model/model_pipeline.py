import os
import boto3
import shutil
import sys
from dotenv import load_dotenv

# --- IMPORTS SECURISES POUR CPU ---
from langchain_groq import ChatGroq
# On utilise la version community pour éviter le crash CPU "Meta Tensor"
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# Configuration
S3_BUCKET_NAME = "g1-data"
S3_ARTIFACT_PATH = "artifacts/vector_index/faiss_index"
LOCAL_INDEX_PATH = "/tmp/faiss_index_g1"

class RAGModel:
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None

    def _download_index_from_s3(self):
        print(f"🔄 Début du téléchargement S3 ({S3_BUCKET_NAME})...")

        if os.path.exists(LOCAL_INDEX_PATH):
            shutil.rmtree(LOCAL_INDEX_PATH)
        os.makedirs(LOCAL_INDEX_PATH)

        try:
            # Client S3 optimisé avec région explicite
            s3 = boto3.client(
                's3', 
                region_name=os.getenv("AWS_REGION", "eu-west-3"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
            )

            files = ["index.faiss", "index.pkl"]
            for file in files:
                source = f"{S3_ARTIFACT_PATH}/{file}"
                destination = f"{LOCAL_INDEX_PATH}/{file}"
                print(f"   📥 Téléchargement de {file}...")
                s3.download_file(S3_BUCKET_NAME, source, destination)

            print("✅ Index FAISS téléchargé.")
        
        except Exception as e:
            raise Exception(f"Erreur S3 : {str(e)}")

    def load_model(self):
        # 1. Téléchargement
        self._download_index_from_s3()

        print("🧠 Chargement des Embeddings (Force CPU)...")
        # 2. Embeddings optimisés CPU
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # 3. Chargement FAISS
        try:
            self.vector_store = FAISS.load_local(
                LOCAL_INDEX_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            raise Exception(f"Erreur lecture FAISS : {e}")

        # 4. LLM & Chain
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key: raise ValueError("GROQ_API_KEY manquante.")

        llm = ChatGroq(
            temperature=0.3, 
            model_name="llama-3.1-8b-instant",
            api_key=api_key
        )

        prompt = ChatPromptTemplate.from_template("""
        Answer based ONLY on the context below.
        <context>{context}</context>
        Question: {input}
        """)

        chain = create_stuff_documents_chain(llm, prompt)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        self.qa_chain = create_retrieval_chain(retriever, chain)

    def predict(self, query):
        if not self.qa_chain: self.load_model()
        response = self.qa_chain.invoke({"input": query})
        sources = [doc.metadata.get('source', 'Inconnu') for doc in response.get('context', [])]
        return response['answer'], sources
