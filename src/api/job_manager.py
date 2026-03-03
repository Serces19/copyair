import asyncio
import subprocess
import os
import signal

class JobManager:
    """
    Singleton class to manage ML scripts (train.py, predict.py) as subprocesses.
    It captures stdout/stderr and stores them for websocket streaming.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(JobManager, cls).__new__(cls)
            cls._instance.current_process = None
            cls._instance.current_job_type = None
            cls._instance.logs = []
            cls._instance.logs_queue = asyncio.Queue()
        return cls._instance

    async def start_job(self, command: list[str], job_type: str):
        if self.current_process is not None and self.current_process.poll() is None:
            raise RuntimeError(f"A job of type '{self.current_job_type}' is already running.")

        self.logs.clear()
        # Limpiamos los logs de la cola asincrona instanciando una nueva
        self.logs_queue = asyncio.Queue()
        self.current_job_type = job_type
        self.log(f"Starting job: {job_type} -> {' '.join(command)}")

        # Arrancamos proceso
        # NOTA: En Windows, la configuración para matar subprocessos con grupos varía,
        # para una implemetación simplificada guardamos el popen.
        kwargs = {}
        if os.name == 'nt':
            kwargs['creationflags'] = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0x00000200)

        self.current_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1, # Line-buffered
            **kwargs
        )
        
        # Leemos en background
        asyncio.create_task(self._log_reader())

    async def _log_reader(self):
        loop = asyncio.get_running_loop()
        while self.current_process and self.current_process.poll() is None:
            # Leer asincronamente línea por línea
            line = await loop.run_in_executor(None, self.current_process.stdout.readline)
            if line:
                self.log(line.strip())
            else:
                break
        
        # Leer el sobrante
        if self.current_process:
            for line in self.current_process.stdout.readlines():
                if line:
                    self.log(line.strip())
            
            exit_code = self.current_process.wait()
            self.log(f"Job {self.current_job_type} finished with exit code {exit_code}")
        
        # Manda token nulo para cerrar listeners
        await self.logs_queue.put(None)

    def log(self, message: str):
        self.logs.append(message)
        # No bloquear el thread de logs si la queue falla
        try:
            self.logs_queue.put_nowait(message)
        except asyncio.QueueFull:
            pass

    def stop_job(self):
        if self.current_process is not None and self.current_process.poll() is None:
            self.log("Stopping job externally...")
            try:
                if os.name == 'nt':
                     self.current_process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    self.current_process.terminate()
            except Exception as e:
                self.log(f"Error terminating process: {e}")
            self.current_process = None
            return True
        return False

    def get_status(self):
        if self.current_process is None:
            return {"status": "idle", "job_type": None}
        poll = self.current_process.poll()
        if poll is None:
            return {"status": "running", "job_type": self.current_job_type}
        return {"status": "finished", "job_type": self.current_job_type, "exit_code": poll}

job_manager = JobManager()
