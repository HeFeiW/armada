# Build stage: Create base environment with CUDA and dependencies
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH \
    PATH=${CUDA_HOME}/bin:$PATH \
    MS2_ASSET_DIR=/workspace/data

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    git \
    ca-certificates \
    software-properties-common \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libopenblas-dev \
    gcc \
    g++ \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.11.0-2-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -afy

ENV PATH=/opt/conda/bin:$PATH

# Create conda environment with ARMADA dependencies
RUN conda config --add channels pytorch && \
    conda config --add channels pytorch3d && \
    conda config --add channels nvidia && \
    conda config --add channels conda-forge

# Install core dependencies via conda
RUN conda create -y -n armada python=3.10 && \
    echo "source activate armada" > ~/.bashrc

SHELL ["/bin/bash", "-c"]
RUN source activate armada && \
    conda install -y \
    pytorch::pytorch=2.1.0 \
    pytorch::torchvision=0.16.0 \
    pytorch::pytorch-cuda=12.1 \
    pytorch3d::pytorch3d=0.7.5 \
    numpy=1.23.5 \
    numba=0.56.4 \
    scipy=1.14.0 \
    opencv=4.6.0 \
    cffi=1.15.1 \
    matplotlib=3.6.1 \
    zarr=2.12.0 \
    numcodecs=0.10.2 \
    h5py=3.7.0 \
    hydra-core=1.3.2 \
    einops=0.4.1 \
    tqdm=4.64.1 \
    dill=0.3.5.1 \
    scikit-video=1.1.11 \
    scikit-image=0.19.3 \
    scipy::gym=0.21.0 \
    pymunk=6.2.1 \
    wandb=0.13.3 \
    threadpoolctl=3.1.0 \
    shapely=1.8.4 \
    cython=0.29.32 \
    imageio=2.22.0 \
    imageio-ffmpeg=0.4.7 \
    termcolor=2.0.1 \
    tensorboard=2.10.1 \
    tensorboardx=2.5.1 \
    psutil=5.9.2 \
    click=8.0.4 \
    boto3=1.24.96 \
    accelerate=0.13.2 \
    datasets=2.6.1 \
    diffusers=0.11.1 \
    av=10.0.0 \
    cmake=3.24.3 \
    llvm-openmp=14 \
    imagecodecs=2022.8.8 \
    ipykernel=6.16

# Install pip dependencies
RUN source activate armada && pip install --no-cache-dir \
    ray==2.2.0 \
    free-mujoco-py==2.1.6 \
    pygame==2.1.2 \
    pybullet-svl==3.1.6.4 \
    robosuite==1.5.1 \
    robomimic==0.2.0 \
    pytorchvideo==0.1.5 \
    imagecodecs==2022.9.26 \
    dm-control==1.0.9 \
    POT==0.9.5 \
    huggingface-hub==0.25.0 \
    kornia==0.8.1

# Install ManiSkill2
RUN source activate armada && pip install --no-cache-dir mani-skill2

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Create data directory for ManiSkill2 assets
RUN mkdir -p /workspace/data && \
    chown -R 1000:1000 /workspace

# Create non-root user
RUN useradd -m -u 1000 armada && \
    chown -R armada:armada /workspace /opt/conda

USER armada

# Default command
CMD ["/bin/bash"]
