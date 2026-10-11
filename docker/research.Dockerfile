FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==2.2.6 scipy==1.15.3 && pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
USER 65534:65534
WORKDIR /tmp
