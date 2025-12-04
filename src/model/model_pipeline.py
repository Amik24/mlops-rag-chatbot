import streamlit as st
import sys
import os

# --- 1. CONFIGURATION DES CHEMINS (Vital pour l'import) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import du backend
try:
    from src.model.model_pipeline import RAGModel
except ImportError as e:
    st.error(f"❌ Erreur d'importation : {e}")
    st.stop()

# --- 2. CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="RAG Course Bot", page_icon="🎓")
st.title("🎓 Assistant de Cours MLOps")

# --- 3. LE SECRET ANTI-CRASH (CACHE) ---
# @st.cache_resource permet de garder le modèle en mémoire
# Cela évite de tout re-télécharger à chaque clic, ce qui cause le Timeout 503
@st.cache_resource(show_spinner=False)
def load_rag_model():
    """Instancie et charge le modèle RAG une seule fois."""
    model = RAGModel()
    model.load_model() # C'est ici que ça prend du temps (Download S3 + HuggingFace)
    return model

# --- 4. CHARGEMENT AVEC FEEDBACK VISUEL ---
if "rag" not in st.session_state:
    # On affiche un message pendant que la fonction cachée travaille
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        status_text.text("⏳ Initialisation : Téléchargement du modèle IA et des données S3...")
        progress_bar.progress(20)
        
        # Appel de la fonction cachée (c'est elle qui fait le travail lourd)
        st.session_state.rag = load_rag_model()
        
        progress_bar.progress(100)
        status_text.success("✅ Assistant prêt ! Posez vos questions.")
        
    except Exception as e:
        status_text.error("❌ Échec du chargement.")
        st.error(f"Erreur technique : {e}")
        st.stop()

# --- 5. INTERFACE DE CHAT ---
st.markdown("---")

# Initialisation de l'historique
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage des anciens messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie
if prompt := st.chat_input("Votre question (ex: 'C'est quoi un Transformer ?')..."):
    # Afficher la question user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Réponse de l'assistant
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyse de vos cours..."):
            try:
                # Prédiction
                answer, sources = st.session_state.rag.predict(prompt)
                
                # Mise en forme de la réponse
                response_text = f"{answer}\n\n"
                if sources:
                    response_text += "---\n**📚 Sources détectées :**\n"
                    # On dédoublonne les sources pour faire propre
                    unique_sources = list(set(sources))
                    for src in unique_sources:
                        response_text += f"- *{src}*\n"
                
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            except Exception as e:
                st.error("Oups, une erreur est survenue.")
                st.write(f"Erreur : {e}")
