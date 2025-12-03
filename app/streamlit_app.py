import streamlit as st
import sys
import os

# --- 1. LE PONT DES SECRETS (CRUCIAL POUR STREAMLIT CLOUD) ---
# Streamlit Cloud stocke les secrets dans `st.secrets`, mais boto3 et LangChain
# cherchent dans `os.environ`. On fait donc le transfert ici.
if "AWS_ACCESS_KEY_ID" in st.secrets:
    os.environ["AWS_ACCESS_KEY_ID"] = st.secrets["AWS_ACCESS_KEY_ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["AWS_SECRET_ACCESS_KEY"]
    os.environ["AWS_SESSION_TOKEN"] = st.secrets["AWS_SESSION_TOKEN"]
    os.environ["AWS_REGION"] = st.secrets["AWS_REGION"]

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# --- 2. CONFIGURATION DES CHEMINS ---
# Permet de trouver les modules 'src' même depuis Streamlit Cloud
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# On importe le modèle SEULEMENT après avoir injecté les secrets
try:
    from src.model.model_pipeline import RAGModel
except ImportError:
    # Fallback pour le développement local si lancé depuis la racine
    from src.model.model_pipeline import RAGModel

# --- 3. CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="RAG Course Bot", page_icon="🎓")

st.title("🎓 Assistant de Cours MLOps")
st.markdown("""
Posez vos questions sur les cours (NLP, RNN, Transformers, etc.).
_L'IA répondra en se basant uniquement sur vos slides de cours._
""")

# --- 4. CHARGEMENT DU MODÈLE (Une seule fois) ---
if "rag" not in st.session_state:
    st.session_state.rag = RAGModel()
    
    with st.spinner("🔄 Téléchargement de l'index depuis S3 et initialisation..."):
        try:
            # C'est ici que le code va chercher 'faiss_index.bin' sur S3
            st.session_state.rag.load_model()
            st.success("✅ Base de connaissance chargée !")
        except Exception as e:
            st.error(f"❌ Erreur critique : Impossible de charger le modèle.")
            st.error(f"Détails : {e}")
            st.info("💡 Avez-vous lancé le Workflow GitHub 'Data Vectorization' pour créer l'index sur S3 ?")

# --- 5. INTERFACE DE CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie utilisateur
if prompt := st.chat_input("Votre question (ex: 'C'est quoi le Vanishing Gradient ?')..."):
    # Afficher la question utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Générer la réponse
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyse des documents..."):
            try:
                answer, sources = st.session_state.rag.predict(prompt)
                
                # Formatage de la réponse avec les sources
                response_text = f"{answer}\n\n---\n**📚 Sources utilisées :**\n"
                for src in sources:
                    response_text += f"- *{src}*\n"
                
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            except Exception as e:
                st.error("Oups, une erreur est survenue lors de la génération.")
                st.write(e)
