import pandas as pd
import numpy as np
import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
args = parser.parse_args()

df = pd.read_csv('getcontact_preprocessing.csv')
df = df.dropna(subset=['text_clean', 'label'])

X = df['text_clean']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Getcontact_Sentiment_CI")

with mlflow.start_run(run_name="CI_Automated_Run"):
    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42)
    mlflow.log_param("n_estimators", args.n_estimators)
    
    model.fit(X_train_vec, y_train)
    
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)
    
    mlflow.sklearn.log_model(model, "model")
    print(f"CI Training Finished successfully. Accuracy: {acc:.4f}")