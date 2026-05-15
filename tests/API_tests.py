import requests
import json

ENDPOINT_API = "https://92t3a8mmij.execute-api.eu-north-1.amazonaws.com/default/mdlops-reviews"

def test_respuesta_positiva():
    payload = json.dumps({
      "review": "This product is amazing, works perfectly!"
    })
    response = requests.request("POST", ENDPOINT_API, data=payload)
    assert response.status_code == 200
    assert response.json()["sentiment"] == "positive"

def test_respuesta_positiva():
    payload = json.dumps({
      "review": "This product is the worst!"
    })
    response = requests.request("POST", ENDPOINT_API, data=payload)
    assert response.status_code == 200
    assert response.json()["sentiment"] == "negative"

def test_campos_vacios():
    payload = json.dumps({
    })
    response = requests.request("POST", ENDPOINT_API, data=payload)
    assert response.status_code == 400

