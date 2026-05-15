FROM public.ecr.aws/lambda/python:3.12

# COPY . ${LAMBDA_TASK_ROOT}
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/lambda_function.py ${LAMBDA_TASK_ROOT}

COPY models/model.pkl      ${LAMBDA_TASK_ROOT}/models/model.pkl
COPY models/vectorizer.pkl ${LAMBDA_TASK_ROOT}/models/vectorizer.pkl

CMD [ "lambda_function.handler" ]