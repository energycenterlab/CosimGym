FROM mcr.microsoft.com/devcontainers/miniconda:0-3

# Remove broken Yarn repository (common in base images)
RUN rm -f /etc/apt/sources.list.d/yarn.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy environment.yml and build the conda environment
COPY environment.yml /tmp/environment.yml
RUN /opt/conda/bin/conda env update -n base -f /tmp/environment.yml && \
    conda clean --all -f -y && \
    rm -f /tmp/environment.yml

# Set PYTHONPATH
ENV PYTHONPATH=/workspace/src
