import joblib
import os


BASE = os.path.dirname(os.path.abspath(__file__))

def test_load_model():
    model = joblib.load(BASE, "models/model.pkl")
    vectorizer = joblib.load(BASE, "models/vectorizer.pkl")
    assert model is not None
    assert vectorizer is not None


def test_predict_labels():
    model = joblib.load(os.path.join(BASE, "..", "models", "model.pkl"))
    vectorizer = joblib.load(os.path.join(BASE, "..", "models", "vectorizer.pkl"))
    vector = vectorizer.transform(["great product"])
    assert model.predict(vector)[0] in ["positive", "negative"]

#Accuracy
def test_accuracy_positive_low():
    model = joblib.load(os.path.join(BASE, "..", "models", "model.pkl"))
    vectorizer = joblib.load(os.path.join(BASE, "..", "models", "vectorizer.pkl"))
    vector = vectorizer.transform(["Yeah this is good!"])
    print(model.predict_proba(vector))
    assert model.predict_proba(vector)[0][1] < 0.70

def test_accuracy_positive_high():
    model = joblib.load(os.path.join(BASE, "..", "models", "model.pkl"))
    vectorizer = joblib.load(os.path.join(BASE, "..", "models","vectorizer.pkl"))
    vector = vectorizer.transform(["Exactly what i wanted"])
    print(model.predict_proba(vector))
    assert model.predict_proba(vector)[0][1] > 0.70


def test_accuracy_negative_low():
    model = joblib.load(os.path.join(BASE, "..", "models", "model.pkl"))
    vectorizer = joblib.load(os.path.join(BASE, "..", "models", "vectorizer.pkl"))
    vector = vectorizer.transform(["it is a bit defective, but works at least"])
    print(model.predict_proba(vector)[0][0])
    assert model.predict_proba(vector)[0][0] < 0.70

def test_accuracy_negative_high():
    model = joblib.load(os.path.join(BASE, "..", "models", "model.pkl"))
    vectorizer = joblib.load(os.path.join(BASE, "..", "models", "vectorizer.pkl"))
    vector = vectorizer.transform(["Totally useless"])
    print(model.predict_proba(vector)[0][0])
    assert model.predict_proba(vector)[0][0] > 0.70

