from fastapi import APIRouter, BackgroundTasks, HTTPException
from src.api.job_manager import job_manager
import os

router = APIRouter()

@router.post("/train")
async def start_training():
    """
    Triggers the training script (scripts/train.py)
    """
    # Usamos .venv\Scripts\python para asegurar el entorno correcto en Windows
    python_exe = os.path.join(".venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = "python" # Fallback a path global

    command = [python_exe, "scripts/train.py", "--config", "configs/params.yaml"]
    
    try:
        await job_manager.start_job(command, "training")
        return {"message": "Training job started", "job_type": "training"}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/predict")
async def start_prediction(model_path: str, video_path: str, output_path: str = "output.mp4"):
    """
    Triggers the inference script (scripts/predict.py)
    """
    python_exe = os.path.join(".venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = "python"

    command = [
        python_exe, "scripts/predict.py", 
        "--model", model_path, 
        "--video", video_path, 
        "--output", output_path,
        "--native-resolution", "--lossless"
    ]
    
    try:
        await job_manager.start_job(command, "inference")
        return {"message": "Inference job started", "job_type": "inference"}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/status")
def get_job_status():
    return job_manager.get_status()

@router.post("/stop")
def stop_job():
    stopped = job_manager.stop_job()
    if stopped:
        return {"message": "Job termination signal sent"}
    return {"message": "No active job to stop"}
