FROM python:3.12-slim

WORKDIR /app

# Install necessary dependencies and the Google Cloud SDK
RUN apt-get update && apt-get install -y \
    gnupg \
    curl \
    apt-transport-https \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Add the Google Cloud SDK distribution URI as a package source
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# Import the Google Cloud public key
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Install the Google Cloud SDK
RUN apt-get update && apt-get install -y google-cloud-sdk && rm -rf /var/lib/apt/lists/*

# Set environment variables
ARG GOOGLE_PROJECT_ID
ARG GCLOUD_SERVICE_KEY

# Copy the service account key file
RUN mkdir -p /tmp
COPY ${GCLOUD_SERVICE_KEY} /tmp/key.json

# Authenticate with Google Cloud and set the project
RUN gcloud auth activate-service-account --key-file=/tmp/key.json && \
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
