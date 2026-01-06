# Use Python 3.10 (Python 3.9+ required)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create outputs directory for volume mounting
RUN mkdir -p /app/outputs

# Set MLflow tracking URI
ENV MLFLOW_TRACKING_URI=file:///app/outputs/mlruns

# Run optimization pipeline
CMD ["python", "src/optimize.py"]
