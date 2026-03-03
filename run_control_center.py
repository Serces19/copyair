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
    print("Iniciando API y servicio UI...")
    
    # Lanzar el navegador en un hilo separado
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Iniciar el servidor Uvicorn
    # Importamos la app aquí para evitar problemas de contexto en hilos
    from src.api.main import app
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
