FROM python:3.12-slim

WORKDIR /app

RUN pip install uvicorn

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py .
COPY pseudoanonymize pseudoanonymize
COPY models models

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
