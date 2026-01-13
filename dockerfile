FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libpython3-dev \
    netcat-openbsd \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*


COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN chmod +x /app/setup_app.sh

CMD ["bash", "/app/setup_app.sh"]