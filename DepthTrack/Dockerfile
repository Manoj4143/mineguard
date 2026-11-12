# CPU runtime image for the dashcam perception pipeline.
# For GPU, base on an nvidia/cuda image and install CUDA torch wheels instead.
FROM python:3.11-slim

# System deps: ffmpeg (encoding) + OpenCV runtime libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first to leverage layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# Install the package itself.
COPY pyproject.toml ./
COPY src ./src
COPY assets ./assets
COPY config ./config
RUN pip install --no-cache-dir .

# Mount your videos at /data and outputs land in /app/output.
ENTRYPOINT ["depthtrack"]
CMD ["--help"]
