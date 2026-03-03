from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os

from src.api.routes import config_routes, job_routes, ws_routes

app = FastAPI(
    title="CopyAir ML Core API",
    description="Local integration API for the CopyAir / NeuralShot Platform backend",
    version="0.1.0",
)

# Configuración de CORS
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar rutas
app.include_router(config_routes.router, prefix="/api/config", tags=["Configuration"])
app.include_router(job_routes.router, prefix="/api/jobs", tags=["Jobs"])
app.include_router(ws_routes.router, prefix="/api/ws", tags=["WebSockets"])

@app.get("/", include_in_schema=False)
def root():
    """Serves the professional test UI"""
    return FileResponse("test_api.html")
