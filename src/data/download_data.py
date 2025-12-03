import os
import boto3
import shutil

S3_BUCKET_NAME = "g1-data"
S3_PREFIX = "raw/" 

def load_data(raw_data_dir="data/raw"):
    """
    Télécharge les fichiers PDF depuis S3 vers le répertoire de données brutes local du runner.
    """
    if os.path.exists(raw_data_dir):
        shutil.rmtree(raw_data_dir)
    os.makedirs(raw_data_dir)

    try:
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=S3_PREFIX)
        files_downloaded = 0
        
        if 'Contents' not in response:
            print(f"❌ Aucun fichier trouvé dans s3://{S3_BUCKET_NAME}/{S3_PREFIX}.")
            return

        for item in response['Contents']:
            key = item['Key']
            if key == S3_PREFIX or not key.lower().endswith(".pdf"):
                continue
                
            file_name = os.path.basename(key)
            local_path = os.path.join(raw_data_dir, file_name)
            s3.download_file(S3_BUCKET_NAME, key, local_path)
            files_downloaded += 1

        print(f"✅ {files_downloaded} files successfully downloaded from S3.")

    except Exception as e:
        raise Exception(f"Erreur critique lors du téléchargement S3: {e}")
