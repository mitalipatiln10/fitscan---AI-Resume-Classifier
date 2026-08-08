FROM python:3.11-slim

WORKDIR /app

# System deps: build tools + LibreOffice (needed for legacy .doc file support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libreoffice \
    && rm -rf /var/lib/apt/lists/*

# Install lightweight CPU-only PyTorch FIRST (default pip torch pulls in
# ~700MB of CUDA/GPU support that's wasted on Render's free tier and eats
# into the 512MB RAM ceiling). This must come before requirements.txt so
# sentence-transformers finds torch already satisfied and doesn't pull the
# full GPU version instead.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Copy the rest of the app
COPY . .

# Render provides the port to use via the $PORT environment variable
EXPOSE 10000
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}
