FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SLT_FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf \
    SLT_HAND_MODEL=/app/models/hand_landmarker.task \
    SLT_NO_DOWNLOAD=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

ADD https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task \
    /app/models/hand_landmarker.task
RUN chmod 0644 /app/models/hand_landmarker.task

COPY app ./app
COPY cli ./cli
COPY tests ./tests
COPY pytest.ini ./

CMD ["python", "-m", "app.main"]