import streamlit as st
import sys
import os

# --- 1. CONFIGURATION DES CHEMINS ---
# Permet de trouver le dossier src/ qui est à la racine
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import du backend (Si ça plante ici, c'est une erreur dans model_pipeline.py)
try:
    from src.model.model_pipeline import RAGModel
except ImportError as e:
    st.error(f"❌ Erreur d'importation : {e}")
    st.stop()

# --- 2. CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="RAG Course Bot", page_icon="🎓")
st.title("🎓 Assistant de Cours MLOps")

# --- 3. LE SECRET ANTI-CRASH (CACHE) ---
# Empêche Streamlit de recharger le modèle à chaque clic (évite le Timeout 503)
@st.cache_resource(show_spinner=False)
def load_rag_model():
    """Instancie et charge le modèle RAG une seule fois."""
    model = RAGModel()
    model.load_model() 
    return model

# --- 4. CHARGEMENT INITIAL ---
if "rag" not in st.session_state:
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        status_text.text("⏳ Initialisation : Téléchargement du modèle IA et S3...")
        progress_bar.progress(20)
        
        # C'est ici que le gros travail se fait
        st.session_state.rag = load_rag_model()
        
        progress_bar.progress(100)
        status_text.success("✅ Assistant prêt ! Posez vos questions.")
        
    except Exception as e:
        status_text.error("❌ Échec du chargement.")
        st.error(f"Erreur technique : {e}")
        st.stop()

# --- 5. CHATBOT ---
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Votre question (ex: 'C'est quoi un Transformer ?')..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyse en cours..."):
            try:
                answer, sources = st.session_state.rag.predict(prompt)
                
                response_text = f"{answer}\n\n"
                if sources:
                    response_text += "---\n**📚 Sources :**\n"
                    unique_sources = list(set(sources))
                    for src in unique_sources:
                        response_text += f"- *{src}*\n"
                
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            except Exception as e:
                st.error("Oups, une erreur est survenue.")
                st.write(f"Erreur : {e}")
