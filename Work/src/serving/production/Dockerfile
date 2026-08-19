# Inference service image.
# The model is mounted at runtime (not baked in) so one image serves many
# model versions - a core 12-factor / model-management practice.

FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app
COPY . /app

ENV PYTHONPATH=/app/Work/src \
    SERVICE_HOST=0.0.0.0 \
    SERVICE_PORT=8000 \
    MODEL_PATH=/models/model.engine \
    MODEL_VERSION=v1

# API key is injected at runtime, never baked into the image.
EXPOSE 8000

CMD ["python", "-m", "serving.production.benchmark"]
