# ✅ Use Python 3.11.9 Slim
FROM python:3.11.9-slim

# ✅ Set working directory
WORKDIR /app

# ✅ Install system dependencies needed for numpy, transformers, torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ✅ Copy project files into container
COPY . /app

# ✅ Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ✅ Expose FastAPI port
EXPOSE 8000

# ✅ Run FastAPI when container starts
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
