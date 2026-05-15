import sys, os
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "..", "data", "dataset_practica.csv")

def load_dataset():
    assert os.path.exists(DATA_PATH), f"No se encuentra el dataset en {DATA_PATH}"


def test_columns():
    df = pd.read_csv(DATA_PATH)
    columnas_esperadas = {"review_body", "language", "sentiment_label", "product_category"}
    assert columnas_esperadas.issubset(set(df.columns)), \
        f"Columnas no encontradas: {df.columns.tolist()}"


def test_empty_dataset():
    df = pd.read_csv(DATA_PATH)
    assert len(df) > 0


def test_null_review_body():
    df = pd.read_csv(DATA_PATH)
    nulos = df["review_body"].isnull().sum()
    assert nulos == 0, f"Hay {nulos} valores nulos en review_body"

def test_null_sentiment_label():
    df = pd.read_csv(DATA_PATH)
    nulos = df["sentiment_label"].isnull().sum()
    assert nulos == 0, f"Hay {nulos} valores nulos en sentiment_label"

def test_null_language():
    df = pd.read_csv(DATA_PATH)
    nulos = df["language"].isnull().sum()
    assert nulos == 0, f"Hay {nulos} valores nulos en language"