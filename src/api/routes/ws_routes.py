from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from src.api.job_manager import job_manager
import asyncio

router = APIRouter()

@router.websocket("/logs")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Al conectar, enviamos los logs acumulados (historial)
    for log in job_manager.logs:
        await websocket.send_text(log)
    
    try:
        # Nos suscribimos a la cola de logs nuevos
        while True:
            # wait_for para evitar bloqueos infinitos si el socket se cuelga
            try:
                message = await asyncio.wait_for(job_manager.logs_queue.get(), timeout=1.0)
                if message is None: # Token de finalización
                    # No cerramos el socket por si el usuario quiere ver el final, 
                    # pero dejamos de esperar en el loop activo.
                    await websocket.send_text("--- JOB FINISHED ---")
                    break
                await websocket.send_text(message)
            except asyncio.TimeoutError:
                # El polling del queue timed out, revisamos si el socket sigue vivo
                continue
                
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Opcional: limpiar recursos o loguear desconexión
        pass
