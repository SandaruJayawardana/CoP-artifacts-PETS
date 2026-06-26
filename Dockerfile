FROM python:3.10-slim

WORKDIR /artifact

# Install basic system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker caching
COPY requirements.txt /artifact/requirements.txt

# Install Python dependencies
RUN python -m pip install --no-cache-dir --upgrade pip
RUN python -m pip install --no-cache-dir -r requirements.txt

# Install notebook support and HiGHS solver interface
RUN python -m pip install --no-cache-dir jupyter nbconvert ipykernel highspy

# Copy the full artifact
COPY . /artifact

# Ensure results folder exists
RUN mkdir -p /artifact/results

CMD ["/bin/bash"]
