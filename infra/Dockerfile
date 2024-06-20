FROM python:3.12-slim

WORKDIR /app

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ARG GOOGLE_PROJECT_ID
ARG GCLOUD_SERVICE_KEY

# Copy the service account key file and configure gcloud
COPY ${GCLOUD_SERVICE_KEY} /tmp/key.json
RUN echo $GCLOUD_SERVICE_KEY | gcloud auth activate-service-account --key-file=/tmp/key.json && \
    gcloud config set project ${GOOGLE_PROJECT_ID}

# Configure pip to use the private repository
RUN mkdir -p ~/.pip && \
    echo "[global]\nindex-url = https://us-east1-python.pkg.dev/ashley-repositories/ashley-python/simple/\n\n[install]\ntrusted-host = us-east1-python.pkg.dev" > ~/.pip/pip.conf

# Install the Python package
RUN pip install ashley_protos==2.3.7

# Install uvicorn
RUN pip install uvicorn

# Copy and install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the application code
COPY main.py .
COPY pseudoanonymize pseudoanonymize
COPY models models

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
