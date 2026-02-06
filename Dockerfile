# Use Ubuntu 22.04 as base image
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV JULIA_VERSION=1.11.2

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    software-properties-common \
    ca-certificates \
    gnupg \
    lsb-release \
    zlib1g-dev \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install Julia 1.11
RUN cd /tmp && \
    wget https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-${JULIA_VERSION}-linux-x86_64.tar.gz && \
    tar -xzf julia-${JULIA_VERSION}-linux-x86_64.tar.gz && \
    mv julia-${JULIA_VERSION} /opt/julia && \
    ln -sf /opt/julia/bin/julia /usr/local/bin/julia && \
    rm julia-${JULIA_VERSION}-linux-x86_64.tar.gz

# Install Julia packages
RUN julia -e 'using Pkg; Pkg.add(["UnitCommitment", "HiGHS", "JuMP"]); Pkg.precompile()'

# Verify Julia installation
RUN julia --version && \
    julia -e "using UnitCommitment, HiGHS, JuMP; println(\"Julia packages OK\")"

# Install PythonCall for Julia-Python bridge
RUN julia -e "using Pkg; Pkg.add(\"PythonCall\"); Pkg.precompile()"

# Create application and data directories
WORKDIR /app
RUN mkdir -p /data/user

# Add labels for documentation
LABEL maintainer="User"
LABEL description="uv and Julia 1.11 development environment"
LABEL julia.version="1.11.2"
LABEL uv.note="Python will be installed via user's pyproject.toml"

# Set default command
CMD ["/bin/bash"]