"""Chequeo rápido de datos: verifica que existan los directorios de input y ground_truth
y que tengan el mismo conjunto de nombres de archivo.

Uso:
    python scripts/check_data.py
"""
from pathlib import Path
import sys
from PIL import Image

BASE = Path(__file__).parent.parent
INPUT_DIR = BASE / 'data' / '03_processed' / 'input'
GT_DIR = BASE / 'data' / '03_processed' / 'ground_truth'


def list_images(p: Path):
    if not p.exists():
        return None
    return sorted([f.name for f in p.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.png')])


def try_open(path: Path):
    try:
        Image.open(path).verify()
        return True
    except Exception:
        return False


def main():
    print(f"Base del proyecto: {BASE}")

    input_list = list_images(INPUT_DIR)
    gt_list = list_images(GT_DIR)

    if input_list is None:
        print(f"ERROR: no existe el directorio de input: {INPUT_DIR}")
        sys.exit(1)
    if gt_list is None:
        print(f"ERROR: no existe el directorio de ground_truth: {GT_DIR}")
        sys.exit(1)

    print(f"Imágenes en input: {len(input_list)}")
    print(f"Imágenes en ground_truth: {len(gt_list)}")

    # comparar nombres
    input_set = set(input_list)
    gt_set = set(gt_list)

    only_input = sorted(list(input_set - gt_set))
    only_gt = sorted(list(gt_set - input_set))

    if only_input:
        print(f"Archivos presentes solo en input ({len(only_input)}): {only_input[:10]}")
    if only_gt:
        print(f"Archivos presentes solo en ground_truth ({len(only_gt)}): {only_gt[:10]}")

    # probar abrir las primeras 3 imágenes de cada carpeta
    for i, name in enumerate(input_list[:3]):
        path = INPUT_DIR / name
        ok = try_open(path)
        print(f"Input {name}: open={'OK' if ok else 'ERROR'}")
    for i, name in enumerate(gt_list[:3]):
        path = GT_DIR / name
        ok = try_open(path)
        print(f"GT {name}: open={'OK' if ok else 'ERROR'}")

    if not only_input and not only_gt:
        print("Filenames coinciden entre input y ground_truth — listo para entrenar.")
    else:
        print("Hay diferencias en los nombres de archivos. Ajusta para que coincidan antes de entrenar.")


if __name__ == '__main__':
    main()
