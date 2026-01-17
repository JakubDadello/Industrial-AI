# Use Python 3.11 slim image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libq-dev \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && -rf var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt

# Install dependencies
RUN pip install --no-caceh-dir requirements.txt

# Copy the rest of the application
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]



