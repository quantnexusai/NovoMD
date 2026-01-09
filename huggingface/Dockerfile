# Hugging Face Spaces Dockerfile for NovoMD
FROM continuumio/miniconda3:latest

WORKDIR /app

# Create conda environment with Python 3.11 and RDKit
RUN conda create -n novomd python=3.11 -y && \
    conda run -n novomd conda install -y -c conda-forge \
    rdkit \
    numpy \
    scipy \
    && conda clean -afy

# Install Gradio in the environment
RUN conda run -n novomd pip install --no-cache-dir gradio>=4.0.0

# Copy application files
COPY app.py .

# Create non-root user for Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user

# Expose Gradio port
EXPOSE 7860

# Set environment variables
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"
ENV GRADIO_MCP_SERVER="True"

# Activate conda env and run (shell form to inherit env vars)
SHELL ["/bin/bash", "-c"]
CMD source activate novomd && python app.py
