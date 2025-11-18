FROM pytorch/pytorch:2.1.0-cuda12.1-runtime-ubuntu22.04

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements y instalar paquetes Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY . .

# Crear directorios necesarios
RUN mkdir -p models output_inference logs

# Comando por defecto: entrenar
CMD ["python", "scripts/train.py", "--config", "configs/params.yaml"]

# Para inferencia: docker run -v /ruta/videos:/app/videos copyair \
#                   python scripts/predict.py --model models/best_model.pth --video videos/input.mp4 --output videos/output.mp4
