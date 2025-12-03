import streamlit as st
import sys
import os
import boto3
from pathlib import Path

# --- IMPORTANT : Préparer le chemin Python pour les imports locaux ---
# Cela permet d'importer la classe RAGModel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.model.model_pipeline import RAGModel

# --- Fonctions Clés pour le Déploiement Cloud ---

def setup_environment():
    """
    Lit les secrets Streamlit et les injecte dans les variables d'environnement.
    CRUCIAL pour que Boto3 et LangChain/Groq trouvent leurs clés.
    """
    if not hasattr(st, 'secrets'):
        st.error("❌ Erreur : Les secrets Streamlit ne sont pas disponibles.")
        st.stop()
        return

    # Configuration AWS pour le téléchargement S3
    try:
        os.environ["AWS_ACCESS_KEY_ID"] = st.secrets["AWS_ACCESS_KEY_ID"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["AWS_SECRET_ACCESS_KEY"]
        # Le token est vital pour les clés SSO temporaires
        os.environ["AWS_SESSION_TOKEN"] = st.secrets["AWS_SESSION_TOKEN"] 
        os.environ["AWS_REGION"] = st.secrets["AWS_REGION"] 
        
        # Configuration Groq (LLM)
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
        
        # Vous pouvez ajouter ici HUGGINGFACEHUB_API_TOKEN si utilisé
        
        print("✅ Variables d'environnement configurées à partir des secrets Streamlit.")
        return True
    except KeyError as e:
        st.error(f"❌ Erreur de configuration de secret : La clé {e} est manquante. Vérifiez vos Secrets Streamlit Cloud.")
        st.stop()
        return False


# --- Configuration et Initialisation Streamlit ---

st.set_page_config(page_title="Course RAG Bot", page_icon="🎓")

st.title("🎓 MLOps Course Assistant")
st.markdown("Posez vos questions sur le NLP, le SVM, les RNNs, et les Transformers, basées sur vos lectures.")

# 1. Configuration de l'environnement au début
if not setup_environment():
    st.stop()


# 2. Initialisation du Modèle (avec téléchargement S3 intégré dans RAGModel.load_model())
if "rag" not in st.session_state:
    st.session_state.rag = RAGModel()
    
    with st.spinner("Chargement de la Base de Connaissances depuis S3..."):
        try:
            # Cette méthode DOIT télécharger l'index S3, puis le charger localement.
            st.session_state.rag.load_model()
            st.success("✅ Modèle RAG chargé !")
        except FileNotFoundError as e:
             st.error(f"❌ Erreur critique : Index FAISS non trouvé. Détail: {e}. Avez-vous exécuté le pipeline de vectorisation CI/CD ?")
             st.session_state.rag.qa_chain = None # Empêche la prédiction
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement du modèle : {e}")
            st.session_state.rag.qa_chain = None # Empêche la prédiction

# 3. Interface de Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher l'historique de chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Gestion de l'entrée utilisateur
if prompt := st.chat_input("Posez votre question ici... (ex: 'Qu'est-ce que le RAG ?'):"):
    # Afficher le message utilisateur
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Générer la réponse
    with st.chat_message("assistant"):
        with st.spinner("Réflexion en cours..."):
            
            # Vérifiez si le modèle a réussi à charger
            if getattr(st.session_state.rag, 'qa_chain', None):
                answer, sources = st.session_state.rag.predict(prompt)
                
                # Formatage des sources
                sources_list = [f"**{src.split('/')[-1]}**" for src in sources]
                
                response_text = f"{answer}\n\n---\n\n📚 **Sources utilisées :** {', '.join(sources_list) if sources_list else 'Aucune source pertinente trouvée.'}"
                st.markdown(response_text)
            else:
                response_text = "Désolé, le modèle n'a pas pu charger. Veuillez vérifier les secrets et le statut du pipeline CI/CD."
                st.error(response_text)
            
    st.session_state.messages.append({"role": "assistant", "content": response_text})
