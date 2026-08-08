FROM python:3.11-slim

WORKDIR /app

# System deps: build tools + LibreOffice (needed for legacy .doc file support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libreoffice \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Copy the rest of the app
COPY . .

# Render provides the port to use via the $PORT environment variable
EXPOSE 10000
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}
