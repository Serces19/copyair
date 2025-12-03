"""
Script de utilidad: Lista todos los modelos disponibles en MLflow
Uso: python scripts/list_mlflow_models.py --mlflow-uri http://tu-servidor:5000
"""

import argparse
import mlflow
from mlflow.tracking import MlflowClient
from tabulate import tabulate
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def list_models(mlflow_uri: str, experiment_name: str, filter_status: str = "FINISHED"):
    """Lista todos los modelos disponibles en MLflow"""
    
    print(f"\n{'='*80}")
    print(f"MLflow URI: {mlflow_uri}")
    print(f"Experimento: {experiment_name}")
    print(f"{'='*80}\n")
    
    # Configurar MLflow
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient(mlflow_uri)
    
    # Obtener experimento
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"❌ Experimento '{experiment_name}' no encontrado")
            
            # Listar experimentos disponibles
            experiments = client.search_experiments()
            print(f"\nExperimentos disponibles:")
            for exp in experiments:
                print(f"  - {exp.name} (ID: {exp.experiment_id})")
            return
    except Exception as e:
        print(f"❌ Error al conectar con MLflow: {e}")
        return
    
    experiment_id = experiment.experiment_id
    print(f"✓ Experimento encontrado (ID: {experiment_id})\n")
    
    # Buscar runs
    filter_string = f"attributes.status = '{filter_status}'" if filter_status else ""
    
    try:
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=filter_string,
            order_by=["start_time DESC"]
        )
    except Exception as e:
        print(f"❌ Error al buscar runs: {e}")
        return
    
    print(f"Total de runs encontrados: {len(runs)}\n")
    
    if not runs:
        print("No hay runs disponibles.")
        return
    
    # Filtrar runs con modelos
    runs_with_models = []
    
    for run in runs:
        run_id = run.info.run_id
        
        try:
            artifacts = client.list_artifacts(run_id)
            has_model = any('checkpoint' in art.path or 'model' in art.path or art.path.endswith('.pth') for art in artifacts)
            
            if has_model:
                params = run.data.params
                metrics = run.data.metrics
                tags = run.data.tags
                
                # Formatear timestamp
                from datetime import datetime
                start_time = datetime.fromtimestamp(run.info.start_time / 1000.0).strftime('%Y-%m-%d %H:%M')
                
                # Extraer métricas clave
                train_loss = metrics.get('train/loss', float('nan'))
                val_psnr = metrics.get('val/psnr', float('nan'))
                val_lpips = metrics.get('val/crop_lpips', float('nan'))
                
                # Extraer parámetros clave
                lr = params.get('training.learning_rate', 'N/A')
                lambda_ssim = params.get('loss.lambda_ssim', 'N/A')
                lambda_perceptual = params.get('loss.lambda_perceptual', 'N/A')
                
                runs_with_models.append({
                    'Run ID': run_id[:8],
                    'Start Time': start_time,
                    'Architecture': params.get('model.architecture', 'N/A'),
                    'Epochs': params.get('training.epochs', 'N/A'),
                    'LR': lr,
                    'Train Loss': f"{train_loss:.4f}" if not float('nan') == train_loss else 'N/A',
                    'Val PSNR': f"{val_psnr:.2f}" if not float('nan') == val_psnr else 'N/A',
                    'Val LPIPS': f"{val_lpips:.4f}" if not float('nan') == val_lpips else 'N/A',
                    'λ_SSIM': lambda_ssim,
                    'λ_Perc': lambda_perceptual,
                })
                
                # Log con más info
                logger.info(f"  ✓ Run: {run_id[:8]} | {start_time} | {params.get('model.architecture', 'N/A')} | Loss: {train_loss:.4f if not float('nan') == train_loss else 'N/A'}")
        except Exception as e:
            print(f"⚠ Error al procesar run {run_id}: {e}")
            continue
    
    if not runs_with_models:
        print("❌ No se encontraron runs con modelos guardados.")
        print("\nVerifica que los modelos se estén guardando correctamente en MLflow.")
        return
    
    # Mostrar tabla
    print(f"✓ Runs con modelos: {len(runs_with_models)}\n")
    print(tabulate(runs_with_models, headers='keys', tablefmt='grid'))
    
    print(f"\n{'='*80}")
    print(f"Para ejecutar batch inference:")
    print(f"  python scripts/batch_inference.py --mlflow-uri {mlflow_uri} --experiment-name {experiment_name}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Lista modelos disponibles en MLflow")
    parser.add_argument('--mlflow-uri', type=str, default='mlruns', help='MLflow tracking URI')
    parser.add_argument('--experiment-name', type=str, default='copyair_experiments', help='Nombre del experimento')
    parser.add_argument('--filter-status', type=str, default='FINISHED', help='Filtrar por estado (FINISHED/RUNNING/FAILED/none)')
    
    args = parser.parse_args()
    
    filter_status = args.filter_status if args.filter_status.lower() != 'none' else None
    
    list_models(args.mlflow_uri, args.experiment_name, filter_status)


if __name__ == '__main__':
    main()
