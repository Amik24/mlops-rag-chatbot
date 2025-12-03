import os
import boto3
import shutil
from dotenv import load_dotenv

# Charge les variables d'environnement (automatique sur GitHub Actions / Streamlit Cloud)
load_dotenv()

# Configuration S3
S3_BUCKET_NAME = "g1-data"
S3_PREFIX = "raw/" 

def load_data(raw_data_dir="data/raw"):
    """
    Télécharge les fichiers PDF depuis S3 (g1-data/raw/) vers le répertoire local 
    pour le traitement par le pipeline.
    """
    # Nettoyage et création du dossier local
    if os.path.exists(raw_data_dir):
        shutil.rmtree(raw_data_dir)
    os.makedirs(raw_data_dir)

    print(f"🔌 Connexion au bucket S3 : {S3_BUCKET_NAME}...")

    try:
        s3 = boto3.client('s3')
        
        # Lister les objets dans le dossier raw/
        response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=S3_PREFIX)
        files_downloaded = 0
        
        if 'Contents' not in response:
            print(f"⚠️ Aucun fichier trouvé dans s3://{S3_BUCKET_NAME}/{S3_PREFIX}")
            return

        for item in response['Contents']:
            key = item['Key']
            # On ignore le dossier lui-même et les fichiers non-pdf
            if key == S3_PREFIX or not key.lower().endswith(".pdf"):
                continue
                
            file_name = os.path.basename(key)
            local_path = os.path.join(raw_data_dir, file_name)

            print(f"⬇️ Téléchargement de {file_name}...")
            s3.download_file(S3_BUCKET_NAME, key, local_path)
            files_downloaded += 1

        print(f"✅ {files_downloaded} fichiers PDF téléchargés dans '{raw_data_dir}'")

    except Exception as e:
        print(f"❌ Erreur critique S3 : {e}")
        # En CI/CD, il faut lever l'erreur pour arrêter le workflow
        raise e

if __name__ == "__main__":
    load_data()
