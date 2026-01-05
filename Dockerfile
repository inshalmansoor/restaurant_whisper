# use a PyTorch image with CUDA support (works well for whisper GPU)
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Avoid buffering in container logs
ENV PYTHONUNBUFFERED=1

# copy code
COPY . /app

# install system deps and pip reqs
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 && \
    pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Expose the port uvicorn will use
EXPOSE 8000

# start uvicorn (single worker recommended for GPU-bound tasks)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--loop", "uvloop"]
