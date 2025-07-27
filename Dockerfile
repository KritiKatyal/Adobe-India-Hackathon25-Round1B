# Use Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY src/ ./src/

# Create directories (not needed as we'll mount volumes)
RUN mkdir -p /app/input /app/output

# Create robust entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
echo "=== Starting PDF Processor ==="\n\
python ./src/process_pdf.py || { echo "PDF processing failed"; exit 1; }\n\
echo "=== Starting Main Analyzer ==="\n\
python ./src/main.py\n\
' > /entrypoint.sh && chmod +x /entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]