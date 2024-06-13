FROM python:3.12-slim

WORKDIR /app

RUN pip install pipenv uvicorn

COPY Pipfile .
COPY Pipfile.lock .

RUN pipenv install

COPY main.py .
COPY pseudoanonymize pseudoanonymize
COPY prompts_examples prompts_examples
EXPOSE 8000

CMD ["pipenv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
