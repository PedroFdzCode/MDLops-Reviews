import argparse
import os
import re
import string
import joblib
import wandb
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEED = 42
np.random.seed(SEED)

WANDB_PROJECT = "amazon-sentiment-mlops"

MODELS = {
    "NaiveBayes":         MultinomialNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED),
    "RandomForest":       RandomForestClassifier(n_estimators=100, random_state=SEED),
}

STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "this", "that", "these",
    "those", "am", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "having", "do", "does", "did", "doing", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "as", "into", "through", "about",
    "not", "no", "nor", "so", "yet", "both", "either", "neither", "than",
    "too", "very", "just", "more", "also", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "each", "every", "any", "few",
    "while", "up", "out", "over", "such", "same", "own", "only", "because"
}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.strip()
    return text


def delete_stop_words(text: str) -> str:
    return " ".join([w for w in text.split() if w not in STOPWORDS])


def preprocess(series: pd.Series) -> pd.Series:
    return series.apply(normalize_text).apply(delete_stop_words)


def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    en_df = df[df["language"] == "en"].reset_index(drop=True)
    print(f"Reviews en inglés cargadas: {len(en_df)}")
    return en_df


def train_model(name: str, model, X_train, X_test, y_train, y_test) -> dict:
    with wandb.init(project=WANDB_PROJECT, name=name, tags=["bow", "sklearn"], reinit=True) as run:

        wandb.config.update({
            "vectorizer": "CountVectorizer",
            "classifier": name,
            "language":   "en",
            "seed":       SEED,
            "test_size":  0.2,
        })

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "f1":        f1_score(y_test, y_pred, pos_label="positive"),
            "precision": precision_score(y_test, y_pred, pos_label="positive"),
            "recall":    recall_score(y_test, y_pred, pos_label="positive"),
        }

        wandb.log(metrics)
        print(f"\n{'='*50}\n  {name}\n{'='*50}")
        print(classification_report(y_test, y_pred))

    return metrics


def save_best_model(best_name: str, best_model, vectorizer, best_metrics: dict):
    models_dir      = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path      = os.path.join(models_dir, "model.pkl")
    vectorizer_path = os.path.join(models_dir, "vectorizer.pkl")

    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Mejor modelo ({best_name}) guardado en {model_path}")

    with wandb.init(project=WANDB_PROJECT, name="best-model-candidate", tags=["candidate", "production"]) as run:

        artifact = wandb.Artifact(
            name="sentiment-classifier",
            type="model",
            description=f"Mejor modelo: {best_name}. Accuracy={best_metrics['accuracy']:.4f}, F1={best_metrics['f1']:.4f}",
            metadata={
                "classifier":    best_name,
                "vectorizer":    "CountVectorizer",
                "language":      "en",
                "val_accuracy":  best_metrics["accuracy"],
                "val_f1":        best_metrics["f1"],
                "val_precision": best_metrics["precision"],
                "val_recall":    best_metrics["recall"],
                "seed":          SEED,
            },
        )

        artifact.add_file(model_path)
        artifact.add_file(vectorizer_path)
        run.log_artifact(artifact)
        wandb.log(best_metrics)

    print("Artefacto 'sentiment-classifier' subido a W&B correctamente.")


def main(data_path: str):
    df = load_data(data_path)
    df["clean_review"] = preprocess(df["review_body"])

    X_text = df["clean_review"]
    y      = df["sentiment_label"]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=SEED
    )

    vectorizer = CountVectorizer()
    X_train    = vectorizer.fit_transform(X_train_text)
    X_test     = vectorizer.transform(X_test_text)

    results = {}
    for name, model in MODELS.items():
        metrics = train_model(name, model, X_train, X_test, y_train, y_test)
        results[name] = {"model": model, "metrics": metrics}

    best_name = max(results, key=lambda n: results[n]["metrics"]["accuracy"])
    best_info = results[best_name]

    print(f"Modelo seleccionado: {best_name} (accuracy={best_info['metrics']['accuracy']:.4f})")

    save_best_model(best_name, best_info["model"], vectorizer, best_info["metrics"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento Amazon Sentiment Analysis")
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "dataset_practica.csv"),
        help="Ruta al CSV del dataset",
    )
    args = parser.parse_args()
    main(args.data_path)
