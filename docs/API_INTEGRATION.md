# API Integration Guide (CopyAir Core)

Esta API sirve como puente entre el motor de entrenamiento de PyTorch y el frontend de ScopeAir. Permite controlar el "Core" sin manipular archivos YAML manualmente.

## Cómo ejecutar la API localmente

1. Asegúrate de tener el entorno activado.
2. Ejecuta el servidor usando `uvicorn`:

```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints Principales

### ⚙️ Configuración (`/api/config`)
- **GET `/`**: Obtiene el contenido actual de `configs/params.yaml`.
- **POST `/`**: Actualiza el `params.yaml`. El frontend envía un JSON y la API lo vuelca al archivo.

### 🚀 Ejecución de Trabajos (`/api/jobs`)
- **POST `/train`**: Lanza `scripts/train.py` usando el intérprete de Python del entorno virtual.
- **POST `/predict`**: Lanza `scripts/predict.py` con argumentos personalizados.
- **GET `/status`**: Indica si hay un proceso corriendo (`idle`, `running`, `finished`).
- **POST `/stop`**: Envía una señal de terminación al proceso activo.

### 📜 Logs en Tiempo Real (`/api/ws`)
- **WS `/logs`**: Canal WebSocket para recibir stdout del entrenamiento línea por línea. Ideal para mostrar la consola en el frontend.

## Estructura de la API
- `src/api/main.py`: Punto de entrada y configuración de CORS.
- `src/api/job_manager.py`: Clase Singleton que gestiona el subproceso, captura logs y los pone en colas asíncronas.
- `src/api/routes/`: Definición de endpoints modularizados.

## Integración con ScopeAir Frontend
En tu frontend de React, puedes conectar al WebSocket así:

```javascript
const socket = new WebSocket('ws://localhost:8000/api/ws/logs');
socket.onmessage = (event) => {
  console.log("Log del entrenamiento:", event.data);
};
```
Y para lanzar el entrenamiento:
```javascript
fetch('http://localhost:8000/api/jobs/train', { method: 'POST' });
```
