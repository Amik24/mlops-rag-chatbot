import pandas as pd
import os
import sys

# Ajouter le chemin 'src' pour les imports (essentiel pour que le runner CI/CD trouve RAGModel)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.model.model_pipeline import RAGModel 

# Le chemin pour sauvegarder le rapport
REPORT_PATH = "data/processed/evaluation_report.csv" 

def evaluate():
    """
    Charge le modèle RAG, exécute un ensemble de questions de test et sauve le rapport.
    """
    
    # 1. Initialisation
    try:
        rag = RAGModel()
        # La méthode load_model() gère le chargement de l'index FAISS depuis le disque local
        rag.load_model()
    except FileNotFoundError as e:
        print(f"❌ Échec de l'évaluation : {e}")
        print("Veuillez d'abord exécuter le pipeline de vectorisation (data-vectorization.yml).")
        return
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle ou des clés : {e}")
        return


    # 2. Cas de Test (Adaptés à vos cours, basés sur les titres des PDF)
    test_cases = [
        "What is the difference between Classic NLP and AI-based NLP?", # Lecture 01
        "Explain the process of Tokenization.", # Lecture 02
        "Why do RNNs suffer from the vanishing gradient problem?", # Lecture 03
        "What is a Support Vector in SVM?", # Lecture 04
        "Define BERT and its main architectural feature.", # Lecture 05
        "What are the limitations of Generative AI in NLP?" # Lecture 06
    ]

    results = []

    print("--- Starting RAG Evaluation ---")
    for question in test_cases:
        print(f"Testing: {question}")
        
        # 3. Prédiction
        answer, sources = rag.predict(question)
        
        # 4. Enregistrement des résultats
        results.append({
            "Question": question,
            "Answer_Preview": answer[:150] + "...", # Prévisualisation de la réponse
            "Sources_Retrieved": ", ".join(sources),
            "Success_Status": "N/A (Manuel)" # Le statut doit être vérifié manuellement
        })

    # 5. Sauvegarde du rapport
    df = pd.DataFrame(results)
    
    # Assurez-vous que le dossier existe
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    
    df.to_csv(REPORT_PATH, index=False)
    
    print("\n" + "="*50)
    print("✅ Évaluation RAG terminée. Rapport sauvegardé.")
    print(f"Rapport : {REPORT_PATH}")
    print("="*50)
    print(df)


if __name__ == "__main__":
    # Assurez-vous que le FAISS index a été créé localement avant de lancer ce script
    evaluate()
