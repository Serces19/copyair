from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from src.api.routes import config_routes, job_routes, ws_routes

app = FastAPI(
    title="CopyAir ML Core API",
    description="Local integration API for the CopyAir / NeuralShot Platform backend",
    version="0.1.0",
)

# Permitir CORS para desarrollo frontend (React/Vite)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción restringir a dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar rutas
app.include_router(config_routes.router, prefix="/api/config", tags=["Configuration"])
app.include_router(job_routes.router, prefix="/api/jobs", tags=["Jobs"])
app.include_router(ws_routes.router, prefix="/api/ws", tags=["WebSockets"])

@app.get("/")
def root():
    return {"message": "Welcome to CopyAir ML API", "status": "online"}
