import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chemin où download_data.py a déposé les fichiers
RAW_DATA_DIR = "data/raw"

def clean_text(text):
    """
    Nettoyage basique pour retirer les headers/footers inutiles des slides de cours.
    """
    # Retire les numéros de page isolés (ex: "PAGE 12")
    text = re.sub(r'PAGE \d+', '', text, flags=re.IGNORECASE)
    # Retire les années académiques ou mentions récurrentes
    text = re.sub(r'202\d-2\d', '', text)
    # Remplace les espaces multiples et sauts de ligne excessifs par un seul espace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_documents(data_dir=RAW_DATA_DIR):
    """
    Charge les PDFs, nettoie le texte et le découpe en chunks.
    Retourne une liste de Documents LangChain.
    """
    documents = []
    
    # Vérification de sécurité
    if not os.path.exists(data_dir):
        print(f"⚠️ Attention : Le dossier {data_dir} n'existe pas.")
        return []

    # 1. Chargement des fichiers PDF
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    
    if not files:
        print(f"⚠️ Aucun fichier PDF trouvé dans {data_dir}")
        return []

    print(f"📄 Traitement de {len(files)} fichiers PDF...")

    for file in files:
        file_path = os.path.join(data_dir, file)
        try:
            # Utilisation du loader optimisé de LangChain
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            # Nettoyage page par page
            for doc in docs:
                doc.page_content = clean_text(doc.page_content)
                # On garde le nom du fichier source pour la citation des sources dans le RAG
                doc.metadata["source_file"] = file 
            
            documents.extend(docs)
            print(f"   ✅ Chargé : {file} ({len(docs)} pages)")
            
        except Exception as e:
            print(f"   ❌ Erreur lors de la lecture de {file}: {e}")

    # 2. Chunking (Segmentation)
    # On utilise une taille de 800 caractères avec un chevauchement de 100
    # C'est un bon équilibre pour capturer le contexte des slides de cours.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ Segmentation terminée : {len(chunks)} chunks générés à partir de {len(documents)} pages.")
    
    return chunks

if __name__ == "__main__":
    # Test local rapide
    chunks = process_documents()
    if chunks:
        print(f"Exemple de chunk : \n{chunks[0].page_content[:200]}...")
