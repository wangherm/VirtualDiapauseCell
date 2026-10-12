FROM python:3.12-slim
# CPU default is portable; GPU wheel choice is explicit, never replace AutoDL host torch.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch==2.8.0 --index-url ${TORCH_INDEX}
RUN pip install --no-cache-dir numpy==2.2.6 scipy==1.15.3 pandas==2.2.3 scikit-learn==1.6.1 \
    h5py==3.13.0 anndata==0.11.4 matplotlib==3.10.1 pyyaml==6.0.2 \
    openpyxl==3.1.5 xlrd==2.0.1 biopython==1.85 pypdf==5.4.0
ENV PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
USER 65534:65534
WORKDIR /tmp
