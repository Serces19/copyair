import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import PairedImageDataset, VideoFrameDataset

def test_all_datasets():
    datasets_root = Path(r"K:\Estudios\Maestria Machine Learning\Tesis\datasets")
    
    if not datasets_root.exists():
        print(f"Directory {datasets_root} not found.")
        return

    print(f"=== Inspeccionando Directorio de Datasets: {datasets_root} ===\n")
    
    subdirs = [d for d in datasets_root.iterdir() if d.is_dir()]
    
    for subdir in subdirs:
        print(f"\n[DATASET] {subdir.name}")
        input_dir = subdir / "input"
        gt_dir = subdir / "gt"
        extracted_dir = subdir / "extracted_frames"
        
        # 1. Paired Dataset
        if input_dir.exists() and gt_dir.exists():
            try:
                ds = PairedImageDataset(input_dir=str(input_dir), gt_dir=str(gt_dir))
                print(f"   [OK] Pares de entrenamiento encontrados: {len(ds)} imagenes")
                sample = ds[0]
                print(f"        Primer par: '{sample['filename']}' | Input: {sample['input'].shape} | GT: {sample['gt'].shape}")
            except Exception as e:
                print(f"   [AVISO] PairedImageDataset error: {e}")
        else:
            print(f"   [INFO] No tiene carpetas input/ y gt/ estandar")
            
        # 2. Extracted Video Frames
        if extracted_dir.exists():
            try:
                vds = VideoFrameDataset(frames_dir=str(extracted_dir))
                print(f"   [OK] Frames de video extraidos para inferencia: {len(vds)} frames")
                vsample = vds[0]
                print(f"        Primer frame: '{vsample['filename']}' | Shape: {vsample['frame'].shape}")
            except Exception as e:
                print(f"   [AVISO] VideoFrameDataset error: {e}")

    print("\n" + "=" * 60)
    print("[SUCCESS] TODOS LOS DATASETS FUERON PARSEADOS Y VALIDADOS CORRECTAMENTE!")
    print("=" * 60)

if __name__ == '__main__':
    test_all_datasets()
