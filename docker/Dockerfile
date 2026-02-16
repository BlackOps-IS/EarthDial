FROM python:3.11-slim

LABEL maintainer="EarthDial <team@earthdial.ai>"
LABEL description="EarthDial — AI Decision Intelligence for Planetary Systems"
LABEL nvidia.tech="Nemotron, cuGraph-compatible, Earth-2 Ecosystem"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Streamlit config
RUN mkdir -p /app/.streamlit
COPY .streamlit/config.toml /app/.streamlit/config.toml

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["python", "-m", "streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.headless=true", \
    "--server.address=0.0.0.0", \
    "--server.enableStaticServing=true"]
