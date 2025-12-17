import os
import boto3
import shutil
import sys
import botocore
from botocore.config import Config
from dotenv import load_dotenv

# --- IMPORTS LANGCHAIN ---
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# --- CONFIGURATION S3 ---
S3_BUCKET_NAME = "g1-data"
S3_ARTIFACT_PATH = "artifacts/vector_index/faiss_index"
LOCAL_INDEX_PATH = "/tmp/faiss_index_g1"

class RAGModel:
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None

    def _download_index_from_s3(self):
        """
        Télécharge l'index depuis S3 en mode ANONYME (Public).
        Cela permet à App Runner de fonctionner sans clés AWS qui expirent.
        """
        print(f"🔄 Téléchargement de l'index depuis S3 ({S3_BUCKET_NAME})...")

        # Nettoyage du dossier temporaire
        if os.path.exists(LOCAL_INDEX_PATH):
            shutil.rmtree(LOCAL_INDEX_PATH)
        os.makedirs(LOCAL_INDEX_PATH)

        try:
            # CONFIGURATION SANS CLÉ (ANONYME)
            # On utilise signature_version=botocore.UNSIGNED pour dire "Pas besoin de clés"
            # Cela nécessite que le bucket S3 soit configuré en accès PUBLIC.
            s3 = boto3.client(
                's3', 
                region_name="eu-west-3",
                config=Config(signature_version=botocore.UNSIGNED)
            )

            files = ["index.faiss", "index.pkl"]
            for file in files:
                source = f"{S3_ARTIFACT_PATH}/{file}"
                destination = f"{LOCAL_INDEX_PATH}/{file}"
                
                print(f"   📥 Téléchargement de {file}...")
                s3.download_file(S3_BUCKET_NAME, source, destination)

            print("✅ Index FAISS téléchargé.")

        except Exception as e:
            raise Exception(f"Erreur S3 (Mode Public) : Impossible de télécharger l'index. \nVérifiez que le bucket '{S3_BUCKET_NAME}' est bien PUBLIC (Permissions S3).\nDétails: {e}")

    def load_model(self):
        # 1. Télécharger les données (Via S3 Public)
        self._download_index_from_s3()

        print("🧠 Chargement des Embeddings (Version Community Stable)...")
        
        # On force le device sur 'cpu'
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # 2. Charger FAISS depuis le dossier temporaire
        try:
            self.vector_store = FAISS.load_local(
                LOCAL_INDEX_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            raise Exception(f"Erreur lors du chargement de FAISS (fichiers corrompus ?): {e}")

        # 3. LLM (Groq)
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY est manquante ! Vérifiez les variables d'environnement App Runner.")

        llm = ChatGroq(
            temperature=0.3, 
            model_name="llama-3.1-8b-instant",
            api_key=api_key
        )

        # 4. Prompt
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant for MLOps students.
        Answer based ONLY on the following context provided.
        If the answer is not in the context, say "I don't know based on the documents".

        <context>
        {context}
        </context>
        
        Question: {input}
        Answer:
        """)

        # 5. Chaîne RAG
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        
        self.qa_chain = create_retrieval_chain(retriever, question_answer_chain)

    def predict(self, query):
        if not self.qa_chain:
            self.load_model()
        
        response = self.qa_chain.invoke({"input": query})
        
        sources = []
        if "context" in response:
            sources = [doc.metadata.get('source', 'Doc inconnu') for doc in response['context']]
            
        return response['answer'], sources

