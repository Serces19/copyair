# CopyAir Frontend

A minimalist, clean web interface for managing training and inference pipelines.

## Features

- **Dashboard**: Overview of system status
- **Training**: Start training, run grid search, view real-time logs
- **Inference**: Upload files, run single or batch inference
- **Settings**: Edit `params.yaml` configuration directly from the UI

## Tech Stack

- **Frontend**: React + TypeScript + Vite
- **Backend**: FastAPI (Python)
- **Styling**: Vanilla CSS (Minimalist Dark Theme)
- **Animations**: GSAP

## Getting Started

### 1. Install Dependencies

```bash
# Backend dependencies (from root directory)
pip install fastapi uvicorn python-multipart

# Frontend dependencies
cd frontend
npm install
```

### 2. Start the Backend

From the **root directory** of the project:

```bash
uvicorn frontend.server:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 3. Start the Frontend

In a **new terminal**, from the `frontend` directory:

```bash
npm run dev
```

The UI will be available at `http://localhost:5173`

## Usage

1. **Configure**: Go to Settings to adjust training parameters
2. **Train**: Start training or run grid search from the Training page
3. **Inference**: Upload images/videos and run inference
4. **Monitor**: View real-time logs and status updates

## API Endpoints

- `GET /api/config` - Get current configuration
- `POST /api/config` - Update configuration
- `POST /api/run/{script_name}` - Run a script (train, predict, etc.)
- `GET /api/status/{script_name}` - Get script status
- `GET /api/logs/{script_name}` - Get script logs
- `POST /api/upload/{folder}` - Upload files to data folder

## Development

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```
