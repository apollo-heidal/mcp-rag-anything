FROM python:3.13-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libreoffice \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py ./
COPY videorag ./videorag
COPY THIRD_PARTY_VIDEORAG_LICENSE ./THIRD_PARTY_VIDEORAG_LICENSE

CMD ["python", "server.py"]
