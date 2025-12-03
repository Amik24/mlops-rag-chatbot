import streamlit as st
import sys
import os
import boto3
from pathlib import Path

# --- Correction de l'importation Locale (Force le chemin de la racine) ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model.model_pipeline import RAGModel 

# --- Fonctions Clés pour le Déploiement Cloud ---

def setup_environment():
    """ Lit les secrets Streamlit et les injecte dans os.environ. """
    # L'application Streamlit va afficher l'erreur détaillée si un secret est manquant.
    if not hasattr(st, 'secrets'):
        return False
        
    try:
        # Clés AWS et Groq
        os.environ["AWS_ACCESS_KEY_ID"] = st.secrets["AWS_ACCESS_KEY_ID"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["AWS_SECRET_ACCESS_KEY"]
        os.environ["AWS_SESSION_TOKEN"] = st.secrets["AWS_SESSION_TOKEN"] 
        os.environ["AWS_REGION"] = st.secrets["AWS_REGION"] 
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
        
        return True
    except KeyError as e:
        st.error(f"❌ Erreur de configuration : Le secret {e} est manquant.")
        return False


# --- Configuration et Initialisation Streamlit ---

st.set_page_config(page_title="Course RAG Bot", page_icon="🎓")

st.title("🎓 MLOps Course Assistant")
st.markdown("Posez vos questions sur le NLP, le SVM, les RNNs, et les Transformers, basées sur vos lectures.")

# 1. Configuration de l'environnement au début
if not setup_environment():
    st.stop()


# 2. Initialisation du Modèle
if "rag" not in st.session_state:
    st.session_state.rag = RAGModel()
    
    with st.spinner("Chargement de la Base de Connaissances depuis S3..."):
        try:
            st.session_state.rag.load_model()
            st.success("✅ Modèle RAG chargé !")
        except FileNotFoundError as e:
             st.error(f"❌ Erreur critique : Index FAISS non trouvé. Avez-vous exécuté le workflow de vectorisation ? Détail: {e}")
             st.session_state.rag.qa_chain = None 
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement du modèle : {e}")
            st.session_state.rag.qa_chain = None 

# 3. Interface de Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher l'historique de chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Gestion de l'entrée utilisateur
if prompt := st.chat_input("Posez votre question ici... (ex: 'Qu'est-ce que le RAG ?'):"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Réflexion en cours..."):
            
            if getattr(st.session_state.rag, 'qa_chain', None):
                answer, sources = st.session_state.rag.predict(prompt)
                
                # Formatage des sources (utilise les résultats de la méthode predict)
                sources_list = [f"**{src.split('/')[-1]}**" for src in sources]
                
                response_text = f"{answer}\n\n---\n\n📚 **Sources utilisées :** {', '.join(sources_list) if sources_list else 'Aucune source pertinente trouvée.'}"
                st.markdown(response_text)
            else:
                response_text = "Désolé, le modèle n'a pas pu charger. Veuillez vérifier les logs d'erreur."
                st.error(response_text)
            
    st.session_state.messages.append({"role": "assistant", "content": response_text})
