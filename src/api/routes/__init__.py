from fastapi import APIRouter
from src.api.routes import config_routes, job_routes, ws_routes

__all__ = ["config_routes", "job_routes", "ws_routes"]
