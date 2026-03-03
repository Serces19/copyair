# NeuralShot Architecture: Edge Node (Vast.ai)

Esta documentación describe cómo el Core de CopyAir funciona como un "Nodo de Ejecución" (Edge Node) dentro de la plataforma NeuralShot.

## Componentes del Nodo
1. **Core Engine (PyTorch):** Los scripts originales `train.py` y `predict.py`.
2. **Integration Layer (FastAPI + WebSocket):** Provee la interfaz REST y el streaming de logs.
3. **Control Center (VFX UI):** Interfaz profesional servida directamente por el nodo para depuración y control manual.

## Flujo de Despliegue en la Nube (Vast.ai)
Para correr este nodo en una instancia remota de GPU:

### 1. Preparación de la Instancia
Se recomienda una imagen Docker con PyTorch precargado. 
- Mapear el puerto `8000` (o el configurado en `PORT`).

### 2. Ejecución Remota
```bash
# Definir variables de entorno de producción
export HOST=0.0.0.0
export PORT=8000
export ALLOWED_ORIGINS=https://tu-plataforma.com,http://localhost:3000
export HEADLESS=1

# Lanzar el servicio
python run_control_center.py
```

### 3. Comunicación con la Plataforma (AWS)
- **AWS Step Functions:** Orquesta la creación de la instancia vía Vast.ai API.
- **WebSocket Gateway:** La plataforma se conecta al WebSocket del nodo para retransmitir los logs al dashboard del usuario.
- **Detección Dinámica:** El Control Center detecta automáticamente la IP de la instancia para todas las operaciones internas.

---
*Status: Cloud Compatible (V1.0.0)*
