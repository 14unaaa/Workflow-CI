import os
import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="getcontact_preprocessing.csv")
    return parser.parse_args()

def upload_to_gdrive(file_path, folder_id):
    """Mengunggah file model ke Google Drive menggunakan Service Account"""
    creds_json = os.environ.get("GDRIVE_SERVICE_ACCOUNT_JSON")
    if not creds_json:
        print("Kredensial GDRIVE_SERVICE_ACCOUNT_JSON tidak ditemukan. Melewati unggah GDrive.")
        return

    with open("gdrive_creds.json", "w") as f:
        f.write(creds_json)

    creds = service_account.Credentials.from_service_account_file(
        "gdrive_creds.json", 
        scopes=["https://www.googleapis.com/auth/drive.file"]
    )
    service = build("drive", "v3", credentials=creds)

    file_metadata = {
        "name": os.path.basename(file_path),
        "parents": [folder_id]
    }
    media = MediaFileUpload(file_path, mimetype="application/octet-stream", resumable=True)
    
    try:
        uploaded_file = service.files().create(body=file_metadata, media_body=media, fields="id").execute()
        print(f"Berhasil mengunggah model ke Google Drive. File ID: {uploaded_file.get('id')}")
    except Exception as e:
        print(f"Gagal mengunggah ke Google Drive: {e}")
    finally:
        if os.path.exists("gdrive_creds.json"):
            os.remove("gdrive_creds.json")

def main():
    args = parse_args()
    
    if not os.path.exists(args.data_path):
        args.data_path = os.path.join("MLProject", os.path.basename(args.data_path))

    print(f"Membaca dataset dari: {args.data_path}")
    df = pd.read_csv(args.data_path)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_test_split=0.2, random_state=42)

    with mlflow.start_run() as run:
        print(f"MLflow Run ID: {run.info.run_id}")
 
        with open("latest_run_id.txt", "w") as f:
            f.write(run.info.run_id)

        n_estimators = 100
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model_artifact")
        
        print(f"Model berhasil dilatih dengan Akurasi: {acc:.4f}")

        local_model_path = "trained_model.pkl"
        import joblib
        joblib.dump(model, local_model_path)

        gdrive_folder_id = os.environ.get("GDRIVE_FOLDER_ID")
        if gdrive_folder_id:
            upload_to_gdrive(local_model_path, gdrive_folder_id)

if __name__ == "__main__":
    main()
