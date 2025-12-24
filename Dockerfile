FROM python:3.12-slim

#working directory 
WORKDIR /app

# Install system dependencies required for Audio (Librosa/SoundFile)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Python packages
RUN pip install --no-cache-dir -r requirements.txt

CMD ["bash"]