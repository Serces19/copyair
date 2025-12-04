import os
import subprocess
import threading
import yaml
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List
import uvicorn
import sys

# Add parent directory to path to import scripts if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = FastAPI(title="CopyAir Frontend API")

@app.get("/")
async def root():
    return {"status": "online", "message": "CopyAir Backend is running"}

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PARAMS_PATH = os.path.join(BASE_DIR, 'configs', 'params.yaml')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Models
class ConfigUpdate(BaseModel):
    config: Dict[str, Any]

class ScriptRun(BaseModel):
    script_name: str
    args: List[str] = []

# Global state for running processes and their logs
running_processes = {}
process_logs = {}

def capture_output(script_name, process):
    """Captures stdout and stderr from a process and appends to a log list."""
    process_logs[script_name] = []
    
    def read_stream(stream, prefix):
        for line in iter(stream.readline, ''):
            process_logs[script_name].append(line)
            # Also print to server console
            print(f"[{script_name}] {line}", end='')
        stream.close()

    t_out = threading.Thread(target=read_stream, args=(process.stdout, ""))
    t_err = threading.Thread(target=read_stream, args=(process.stderr, ""))
    
    t_out.start()
    t_err.start()
    
    t_out.join()
    t_err.join()
    process.wait()

@app.get("/api/config")
async def get_config():
    try:
        with open(PARAMS_PATH, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/config")
async def update_config(update: ConfigUpdate):
    try:
        with open(PARAMS_PATH, 'w') as f:
            yaml.dump(update.config, f, default_flow_style=False)
        return {"status": "success", "message": "Configuration updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/run/{script_name}")
async def run_script(script_name: str, run_req: ScriptRun):
    script_path = os.path.join(BASE_DIR, 'scripts', f"{script_name}.py")
    if not os.path.exists(script_path):
        raise HTTPException(status_code=404, detail="Script not found")
    
    if script_name in running_processes and running_processes[script_name].poll() is None:
        raise HTTPException(status_code=400, detail="Script already running")

    try:
        # Run script in a separate process
        cmd = [sys.executable, script_path] + run_req.args
        # Unbuffered output for real-time logging
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=BASE_DIR,
            env=env,
            bufsize=1, 
            universal_newlines=True
        )
        running_processes[script_name] = process
        
        # Start capturing logs in background
        thread = threading.Thread(target=capture_output, args=(script_name, process))
        thread.daemon = True
        thread.start()
        
        return {"status": "started", "pid": process.pid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status/{script_name}")
async def get_status(script_name: str):
    status = "stopped"
    if script_name in running_processes:
        process = running_processes[script_name]
        if process.poll() is None:
            status = "running"
        else:
            status = "finished"
            
    return {"status": status}

@app.get("/api/logs/{script_name}")
async def get_logs(script_name: str):
    if script_name not in process_logs:
        return {"logs": []}
    return {"logs": process_logs[script_name]}

@app.post("/api/upload/{folder}")
async def upload_file(folder: str, file: UploadFile = File(...)):
    target_dir = os.path.join(DATA_DIR, folder)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    
    file_path = os.path.join(target_dir, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
