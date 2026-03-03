from fastapi import APIRouter, HTTPException
import yaml
import os

router = APIRouter()

CONFIG_PATH = "configs/params.yaml"

@router.get("/")
def get_config():
    if not os.path.exists(CONFIG_PATH):
        raise HTTPException(status_code=404, detail="Config file not found")
    
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading config: {str(e)}")

@router.post("/")
def update_config(new_config: dict):
    """
    Updates the params.yaml file with the provided dictionary.
    Note: This overwrites the file.
    """
    try:
        # Asegurarse de que el directorio existe
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)
            
        return {"message": "Configuration updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating config: {str(e)}")

# Aliasing for the router import in main.py
router = router
