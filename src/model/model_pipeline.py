# --- DÉBUT DU BLOC DE DEBUG S3 ---
import boto3
import streamlit as st
import os

st.divider()
st.subheader("🕵️‍♂️ Diagnostic S3 en direct")

try:
    # 1. Test de connexion
    s3_test = boto3.client(
        's3',
        region_name=st.secrets["AWS_REGION"],
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
    )
    
    # 2. Qui suis-je ?
    identity = boto3.client(
        'sts',
        region_name=st.secrets["AWS_REGION"],
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
    ).get_caller_identity()
    st.write(f"🔑 Connecté en tant que : `{identity['Arn']}`")

    # 3. Liste des buckets visibles
    st.write("📦 Buckets visibles par ces clés :")
    buckets = s3_test.list_buckets()
    found_g1 = False
    for b in buckets['Buckets']:
        st.code(f"- {b['Name']}")
        if b['Name'] == "g1-data":
            found_g1 = True
    
    if not found_g1:
        st.error("❌ Le bucket 'g1-data' n'est PAS dans la liste ! Vérifiez le nom ou les droits.")
    else:
        st.success("✅ Le bucket 'g1-data' est bien visible.")
        
        # 4. Vérification du fichier spécifique
        # ATTENTION : Vérifiez que ce chemin correspond à votre structure S3
        prefix = "artifacts/vector_index/faiss_index/" 
        st.write(f"📂 Recherche de fichiers dans : `g1-data/{prefix}`")
        
        objects = s3_test.list_objects_v2(Bucket="g1-data", Prefix=prefix)
        if 'Contents' in objects:
            for obj in objects['Contents']:
                st.write(f"   📄 Trouvé : `{obj['Key']}` (Taille: {obj['Size']} bytes)")
        else:
            st.error(f"❌ Aucun fichier trouvé dans '{prefix}'. Le dossier est vide ou le chemin est faux.")

except Exception as e:
    st.error(f"💥 Erreur lors du diagnostic : {e}")

st.divider()
# --- FIN DU BLOC DE DEBUG S3 ---
import os
import boto3
import shutil
import sys
from dotenv import load_dotenv

# --- IMPORTS ---
from langchain_groq import ChatGroq
# ⚠️ IMPORTANT : On utilise 'community' pour la stabilité CPU sur Streamlit Cloud
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# --- CONFIGURATION ---
# ⚠️ ATTENTION : Remplace 'g1-data' par LE VRAI NOM de ton bucket S3
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

        # Nettoyage du dossier local
        if os.path.exists(LOCAL_INDEX_PATH):
            shutil.rmtree(LOCAL_INDEX_PATH)
        os.makedirs(LOCAL_INDEX_PATH)

        try:
            # Client S3 avec région explicite (Vital pour éviter l'erreur 400)
            region = os.getenv("AWS_REGION", "eu-west-3")
            s3 = boto3.client(
                's3', 
                region_name=region,
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
            current_region = os.getenv('AWS_REGION')
            raise Exception(f"Erreur S3 (Region: {current_region}) : Impossible de télécharger l'index. \nVérifiez le nom du bucket '{S3_BUCKET_NAME}' et vos droits d'accès.\nDétails: {e}")

    def load_model(self):
        # 1. Télécharger les données
        self._download_index_from_s3()

        print("🧠 Chargement des Embeddings (Force CPU)...")
        # On utilise le device 'cpu' pour éviter le crash sur Streamlit Cloud
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # 2. Charger FAISS
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
            raise ValueError("GROQ_API_KEY est manquante ! Vérifie tes 'Secrets' Streamlit.")

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

