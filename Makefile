# Makefile para CopyAir

.PHONY: help install clean train test predict docker-build docker-run

help:
	@echo "CopyAir - Image-to-Image Translation"
	@echo ""
	@echo "Comandos disponibles:"
	@echo "  make install          - Instalar dependencias"
	@echo "  make train            - Entrenar modelo"
	@echo "  make test             - Ejecutar pruebas"
	@echo "  make predict          - Inferencia en video"
	@echo "  make clean            - Limpiar archivos temporales"
	@echo "  make docker-build     - Construir imagen Docker"
	@echo "  make docker-run       - Ejecutar en Docker"
	@echo "  make lint             - Verificar código"

install:
	pip install -r requirements.txt
	echo "✓ Dependencias instaladas"

train:
	python scripts/train.py --config configs/params.yaml --device cuda

train-cpu:
	python scripts/train.py --config configs/params.yaml --device cpu

predict:
	@read -p "Ruta modelo: " MODEL; \
	read -p "Video entrada: " VIDEO; \
	read -p "Video salida: " OUTPUT; \
	python scripts/predict.py --model $$MODEL --video $$VIDEO --output $$OUTPUT

test:
	pytest tests/ -v --tb=short

test-coverage:
	pytest tests/ --cov=src --cov-report=html

lint:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/
	flake8 src/ scripts/ tests/ --max-line-length=100

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov

docker-build:
	docker build -t copyair:latest .

docker-run:
	docker run --gpus all --rm -v $(PWD)/data:/app/data \
	  -v $(PWD)/models:/app/models \
	  -v $(PWD)/output_inference:/app/output_inference \
	  copyair:latest

setup: clean install
	mkdir -p data/01_raw data/02_interim data/03_processed/input data/03_processed/ground_truth
	mkdir -p models output_inference logs
	echo "✓ Proyecto configurado"

freeze-deps:
	pip freeze > requirements.lock
	echo "✓ Dependencias congeladas"
