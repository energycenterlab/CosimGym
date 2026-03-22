FROM mcr.microsoft.com/devcontainers/miniconda:0-3

# Remove broken Yarn repository (common in base images)
RUN rm -f /etc/apt/sources.list.d/yarn.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libstdc++6 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Initialize conda for all shells
RUN conda init bash

# Copy environment.yml and build the conda environment
COPY environment.yml /tmp/environment.yml
RUN /opt/conda/bin/conda env create -f /tmp/environment.yml && \
    conda clean --all -f -y && \
    rm -f /tmp/environment.yml

# Set PYTHONPATH and library paths
ENV PYTHONPATH=/workspace/src
ENV LD_LIBRARY_PATH=/opt/conda/envs/cosim_gym/lib:$LD_LIBRARY_PATH

# Activate cosim_gym environment by default
RUN echo "conda activate cosim_gym" >> /root/.bashrc
