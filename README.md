# Amazon Reviews Sentiment Analysis - MLOps

**Autor:** Pedro Fernández Sánchez

Proyecto de MLOps desarrollado para el Máster de Deep Learning de la Universidad Politécnica de Madrid. Se aplican las metodologías y herramientas de MLOps sobre un modelo de NLP que clasifica reviews de Amazon en **positive** o **negative**.
## Estructura del repositorio

```
MLOps/
├── data/
│   └── dataset_practica.csv
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
├── src/
│   ├── train.py
│   └── lambda_function.py
├── tests/
│   ├── API-tests.py
│   ├── test_unitario.py
│   ├── test_datos.py
│   └── test_modelo.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile
├── requirements.txt
└── README.md
```
---

## Requisitos

- Python 3.11
- Docker
- Cuenta en Weights & Biases
- AWS CLI configurado (para el despliegue)

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## Cómo entrenar el modelo

Configura tu API key de W&B en un fichero `.env` en la raíz del proyecto:

```
WANDB_API_KEY=tu_clave_aqui
```

Ejecuta el entrenamiento:

```bash
python src/train.py
```

Esto entrenará los 3 modelos, los comparará en W&B y guardará el mejor como artefacto en `models/`.

---

## Cómo construir y lanzar Docker en local

```bash
docker build -t mdlops-reviews .
docker run -p 9000:8080 mdlops-reviews
```

**Llamada en local**
```bash
Invoke-WebRequest -Method POST -Uri "http://localhost:9000/2015-03-31/functions/function/invocations" -ContentType "application/json" -Body '{"body": "{\"review\": \"This product is amazing!\"}"}'
```

---

## Cómo llamar a la API

**Endpoint en producción:**

```bash
curl --location 'https://92t3a8mmij.execute-api.eu-north-1.amazonaws.com/default/mdlops-reviews' \
--header 'Content-Type: application/json' \
--data '{
    "review": "it is a bit defective, but works at least"
}'
```

**Respuesta esperada:**

```json
{
    "sentiment": "negative",
    "confidence": 0.5875
}
```
---

## Cómo ejecutar los tests

```bash
pytest tests/ -v
```

Los tests de API se ejecutan de forma manual ya que requieren el endpoint en producción:

```bash
python -m pytest tests/API_tests.py -v
```

---

## CI/CD

El proyecto tiene integración continua configurada con GitHub Actions. En cada push a `main` se ejecutan automáticamente los tests unitarios, de datos y del modelo.

---
## Links

- **GitHub:** https://github.com/PedroFdzCode/MDLops-Reviews
- **Weights & Biases:** https://api.wandb.ai/links/pedrofdezs01-universidad-polit-cnica-de-madrid/aluf6qjw
- **Endpoint AWS:** https://92t3a8mmij.execute-api.eu-north-1.amazonaws.com/default/mdlops-reviews
