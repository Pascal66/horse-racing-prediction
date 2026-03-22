# Use Python 3.12 (Compatible with Pandas >= 2.2.3)
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# build-essential helps if any pip package needs to compile C code
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
# (Docker will skip this step if requirements.txt hasn't changed)
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project to ensure packages (backend, frontend) are available
COPY . .

# Set PYTHONPATH so imports like "from src.api..." work everywhere
ENV PYTHONPATH=/app

# Document that the container listens on port 8000
EXPOSE 8000

# Start the application using the full package path
CMD ["uvicorn", "backend.src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]