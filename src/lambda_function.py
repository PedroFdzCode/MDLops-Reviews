import json, os, re, string, joblib

# Los modelos están en la imagen, carga directa
BASE = os.path.dirname(os.path.abspath(__file__))
model      = joblib.load(os.path.join(BASE, "models", "model.pkl"))
vectorizer = joblib.load(os.path.join(BASE, "models", "vectorizer.pkl"))

STOPWORDS = {
    "i", "me", "my", "we", "our", "you", "your", "he", "she",
    "it", "they", "them", "what", "which", "this", "that", "am",
    "is", "are", "was", "were", "be", "been", "have", "has", "had",
    "do", "does", "did", "a", "an", "the", "and", "or", "but",
    "in", "on", "at", "to", "for", "of", "with", "by", "from"
}

def handler(event, context):
    try:
        body   = json.loads(event.get("body", "{}"))
        review = body.get("review", "").strip()

        if not review:
            return {"statusCode": 400,
                    "body": json.dumps({"error": "El campo 'review' es obligatorio"})}

        vector     = vectorizer.transform([preprocess(review)])
        prediction = model.predict(vector)[0]
        confidence = round(float(max(model.predict_proba(vector)[0])), 4)

        return {"statusCode": 200,
                "body": json.dumps({"sentiment": prediction, "confidence": confidence})}

    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}

def preprocess(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join([w for w in text.split() if w not in STOPWORDS])
