import sys, os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.train import preprocess

def test_minusculas():
    text = "EXAMPLE TEXT"
    resultado = preprocess(pd.Series([text]))
    assert resultado[0] == text.lower()

def test_eliminar_puntuacion():
    text = "example, text!"
    resultado = preprocess(pd.Series([text]))
    assert "!" not in resultado[0]
    assert "," not in resultado[0]

def test_eliminar_numeros():
    text = "This is a test 123"
    resultado = preprocess(pd.Series([text]))
    assert "123" is not resultado[0]

def test_elimina_stopwords():
    text = "this is a great product"
    resultado = preprocess(pd.Series([text]))
    assert "this" not in resultado[0]
