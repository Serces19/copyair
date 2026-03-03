import uvicorn
import webbrowser
import threading
import time
import os
import sys

# Asegurar que el directorio raíz está en el path para imports de src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def open_browser():
    """Espera un momento a que el servidor inicie y abre el navegador."""
    time.sleep(2)
    print("\n🚀 Lanzando interfaz de Control Center...")
    webbrowser.open("http://localhost:8000/")

if __name__ == "__main__":
    print("--- CopyAir / NeuralShot Control Center ---")
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    # Solo abrir navegador si estamos en localhost o se solicita explícitamente
    if host in ["127.0.0.1", "localhost", "0.0.0.0"] and os.getenv("HEADLESS") != "1":
        print(f"Iniciando API y servicio UI en http://{host}:{port}...")
        threading.Thread(target=open_browser, daemon=True).start()
    else:
        print(f"Servidor iniciado en modo remoto/headless en http://{host}:{port}")
    
    # Iniciar el servidor Uvicorn
    from src.api.main import app
    uvicorn.run(app, host=host, port=port, log_level="info")
