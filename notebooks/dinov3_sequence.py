import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Importar las clases del script original
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Copiar las clases necesarias
class DINOv3Embedder:
    """Clase para extraer embeddings usando DINOv3"""
    
    CANDIDATES = ['facebook/dinov2-base', 'facebook/dinov3-vitb16-pretrain-lvd1689m']
    
    def __init__(self,
                 model_name=None,
                 device='cuda',
                 input_size=224,
                 pooling='mean',
                 use_fp16=True):
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        assert pooling in ('tokens', 'mean', 'cls'), "pooling must be 'tokens','mean' or 'cls'"
        self.pooling = pooling
        self.use_fp16 = use_fp16

        # Determinar modelo a cargar
        candidates = [model_name] + [m for m in self.CANDIDATES if m != model_name] if model_name else self.CANDIDATES

        load_err = None
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                print(f"[DINOv3] Cargando modelo: {candidate}")
                from transformers import AutoImageProcessor, AutoModel
                self.processor = AutoImageProcessor.from_pretrained(candidate)
                self.model = AutoModel.from_pretrained(candidate).to(self.device)
                self.model_name = candidate
                
                if self.use_fp16:
                    self.model = self.model.half()
                
                self.model.eval()
                print(f"[DINOv3] Modelo cargado exitosamente en {self.device}")
                break
                
            except Exception as e:
                load_err = e
                print(f"[DINOv3] Error cargando {candidate}: {e}")
                continue
        else:
            raise RuntimeError(f"No se pudo cargar ningún modelo DINO. Último error: {load_err}")
        
        # Transformaciones para imágenes
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def get_embedding(self, image_tensor):
        """Extraer embedding de un tensor de imagen"""
        with torch.no_grad():
            if self.use_fp16:
                image_tensor = image_tensor.half()
            
            image_tensor = image_tensor.to(self.device)
            
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            # Obtener características del modelo
            outputs = self.model(image_tensor)
            
            # Procesar según el tipo de pooling
            if self.pooling == 'cls':
                features = outputs.last_hidden_state[:, 0, :]
            elif self.pooling == 'mean':
                features = outputs.last_hidden_state.mean(dim=1)
            elif self.pooling == 'tokens':
                features = outputs.last_hidden_state.flatten(1)
            
            # Normalizar embeddings
            features = features / features.norm(dim=-1, keepdim=True)
            
            return features.cpu().numpy()
    
    def process_image(self, image_path):
        """Procesar imagen desde ruta"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)
        return image_tensor, image


class SequenceAnalyzer:
    """Analizador de embeddings para secuencias de imágenes"""
    
    def __init__(self, model_name='facebook/dinov3-vitb16-pretrain-lvd1689m'):
        self.embedder = DINOv3Embedder(model_name=model_name, pooling='mean')
    
    def analyze_sequence(self, sequence_dir, frame_step=5, max_frames=None, reference_frame=None):
        """Analizar secuencia de imágenes
        
        Args:
            sequence_dir: Directorio con la secuencia
            frame_step: Extraer 1 de cada N frames
            max_frames: Máximo número de frames a procesar
            reference_frame: Path a frame de referencia (opcional)
        """
        print(f"\n{'='*60}")
        print(f"ANALIZANDO SECUENCIA: {sequence_dir}")
        print(f"{'='*60}")
        
        # Buscar todas las imágenes
        sequence_path = Path(sequence_dir)
        image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
        
        all_images = []
        for ext in image_extensions:
            all_images.extend(sorted(sequence_path.glob(f'*{ext}')))
        
        if not all_images:
            raise ValueError(f"No se encontraron imágenes en {sequence_dir}")
        
        # Filtrar por frame_step
        selected_frames = all_images[::frame_step]
        
        if max_frames is not None:
            selected_frames = selected_frames[:max_frames]
        
        print(f"Total de imágenes en secuencia: {len(all_images)}")
        print(f"Frames seleccionados (cada {frame_step}): {len(selected_frames)}")
        
        # Procesar frame de referencia si existe
        reference_embedding = None
        if reference_frame:
            print(f"\nProcesando frame de referencia: {reference_frame}")
            ref_tensor, _ = self.embedder.process_image(reference_frame)
            reference_embedding = self.embedder.get_embedding(ref_tensor)
        
        # Extraer embeddings de la secuencia
        embeddings = []
        metadata = []
        
        for idx, frame_path in enumerate(selected_frames):
            # Extraer número de frame
            frame_num = int(''.join(filter(str.isdigit, frame_path.stem)))
            
            # Procesar imagen
            tensor, _ = self.embedder.process_image(str(frame_path))
            embedding = self.embedder.get_embedding(tensor)
            
            embeddings.append(embedding)
            metadata.append({
                'frame_number': frame_num,
                'frame_index': idx,
                'path': str(frame_path),
                'is_reference': (reference_frame and str(frame_path) == reference_frame)
            })
            
            if (idx + 1) % 10 == 0:
                print(f"  Procesados {idx + 1}/{len(selected_frames)} frames...")
        
        embeddings = np.vstack(embeddings)
        
        print(f"\n✓ Embeddings extraídos: {embeddings.shape}")
        
        return embeddings, metadata, reference_embedding
    
    def visualize_sequence(self, embeddings, metadata, reference_embedding=None):
        """Visualizar embeddings de secuencia en 2D y 3D"""
        
        print("\n" + "="*60)
        print("REDUCCIÓN DIMENSIONAL Y VISUALIZACIÓN")
        print("="*60)
        
        # Incluir referencia si existe
        if reference_embedding is not None:
            embeddings_with_ref = np.vstack([reference_embedding, embeddings])
            ref_idx = 0
        else:
            embeddings_with_ref = embeddings
            ref_idx = None
        
        # Frame numbers para colorear
        frame_numbers = np.array([m['frame_number'] for m in metadata])
        
        # ============================================================
        # PCA
        # ============================================================
        print("\n[1/3] Aplicando PCA...")
        pca_2d = PCA(n_components=2, random_state=42)
        embeddings_pca_2d = pca_2d.fit_transform(embeddings_with_ref)
        variance_2d = pca_2d.explained_variance_ratio_
        print(f"  → Varianza explicada: PC1={variance_2d[0]:.2%}, PC2={variance_2d[1]:.2%}, Total={variance_2d.sum():.2%}")
        
        pca_3d = PCA(n_components=3, random_state=42)
        embeddings_pca_3d = pca_3d.fit_transform(embeddings_with_ref)
        variance_3d = pca_3d.explained_variance_ratio_
        print(f"  → Varianza 3D: Total={variance_3d.sum():.2%}")
        
        # ============================================================
        # t-SNE
        # ============================================================
        print("\n[2/3] Aplicando t-SNE...")
        pca_50 = PCA(n_components=min(50, embeddings_with_ref.shape[0]-1), random_state=42)
        embeddings_pca_50 = pca_50.fit_transform(embeddings_with_ref)
        
        perplexity = min(30, embeddings_with_ref.shape[0] // 3)
        tsne_2d = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        embeddings_tsne_2d = tsne_2d.fit_transform(embeddings_pca_50)
        
        tsne_3d = TSNE(n_components=3, perplexity=perplexity, random_state=42, n_iter=1000)
        embeddings_tsne_3d = tsne_3d.fit_transform(embeddings_pca_50)
        print(f"  → Perplexity: {perplexity}")
        
        # ============================================================
        # UMAP (opcional)
        # ============================================================
        print("\n[3/3] Aplicando UMAP...")
        try:
            from umap import UMAP
            n_neighbors = min(15, embeddings_with_ref.shape[0] // 2)
            umap_2d = UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42, min_dist=0.1)
            embeddings_umap_2d = umap_2d.fit_transform(embeddings_with_ref)
            
            umap_3d = UMAP(n_components=3, n_neighbors=n_neighbors, random_state=42, min_dist=0.1)
            embeddings_umap_3d = umap_3d.fit_transform(embeddings_with_ref)
            print(f"  → n_neighbors: {n_neighbors}")
            has_umap = True
        except ImportError:
            print("  ⚠ UMAP no disponible (instalar con: pip install umap-learn)")
            has_umap = False
        
        # ============================================================
        # VISUALIZACIONES 2D
        # ============================================================
        n_methods = 3 if has_umap else 2
        fig_2d, axes = plt.subplots(2, n_methods, figsize=(6*n_methods, 12))
        
        methods_2d = [
            ('PCA', embeddings_pca_2d, f'Varianza: {variance_2d.sum():.1%}'),
            ('t-SNE', embeddings_tsne_2d, f'Perplexity: {perplexity}'),
        ]
        if has_umap:
            methods_2d.append(('UMAP', embeddings_umap_2d, f'n_neighbors: {n_neighbors}'))
        
        for col, (method_name, coords_2d, subtitle) in enumerate(methods_2d):
            # Fila 1: Secuencia coloreada por número de frame
            ax = axes[0, col]
            
            if ref_idx is not None:
                # Plotear secuencia (sin referencia)
                scatter = ax.scatter(coords_2d[1:, 0], coords_2d[1:, 1],
                                   c=frame_numbers, cmap='viridis', alpha=0.7, s=40)
                # Marcar referencia
                ax.scatter(coords_2d[ref_idx, 0], coords_2d[ref_idx, 1],
                          c='red', s=300, marker='*', label='Referencia',
                          edgecolors='black', linewidth=2, zorder=10)
            else:
                scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1],
                                   c=frame_numbers, cmap='viridis', alpha=0.7, s=40)
            
            ax.set_title(f'{method_name} - Secuencia temporal\n{subtitle}', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Dimensión 1')
            ax.set_ylabel('Dimensión 2')
            plt.colorbar(scatter, ax=ax, label='Número de frame')
            if ref_idx is not None:
                ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Fila 2: Trayectoria conectada
            ax = axes[1, col]
            
            if ref_idx is not None:
                seq_coords = coords_2d[1:]
            else:
                seq_coords = coords_2d
            
            # Dibujar línea conectando frames consecutivos
            ax.plot(seq_coords[:, 0], seq_coords[:, 1], 
                   'gray', alpha=0.3, linewidth=1, zorder=1)
            
            # Plotear puntos
            scatter = ax.scatter(seq_coords[:, 0], seq_coords[:, 1],
                               c=frame_numbers, cmap='viridis', alpha=0.7, s=40, zorder=2)
            
            # Marcar inicio y fin
            ax.scatter(seq_coords[0, 0], seq_coords[0, 1],
                      c='green', s=200, marker='o', label='Inicio',
                      edgecolors='black', linewidth=2, zorder=3)
            ax.scatter(seq_coords[-1, 0], seq_coords[-1, 1],
                      c='red', s=200, marker='s', label='Fin',
                      edgecolors='black', linewidth=2, zorder=3)
            
            if ref_idx is not None:
                ax.scatter(coords_2d[ref_idx, 0], coords_2d[ref_idx, 1],
                          c='orange', s=300, marker='*', label='Referencia',
                          edgecolors='black', linewidth=2, zorder=10)
            
            ax.set_title(f'{method_name} - Trayectoria temporal\n{subtitle}', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Dimensión 1')
            ax.set_ylabel('Dimensión 2')
            plt.colorbar(scatter, ax=ax, label='Número de frame')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        fig_2d.suptitle('Análisis de Secuencia de Imágenes (2D)', 
                       fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # ============================================================
        # VISUALIZACIONES 3D
        # ============================================================
        fig_3d = plt.figure(figsize=(6*n_methods, 6))
        
        methods_3d = [
            ('PCA 3D', embeddings_pca_3d, f'Varianza: {variance_3d.sum():.1%}'),
            ('t-SNE 3D', embeddings_tsne_3d, f'Perplexity: {perplexity}'),
        ]
        if has_umap:
            methods_3d.append(('UMAP 3D', embeddings_umap_3d, f'n_neighbors: {n_neighbors}'))
        
        for col, (method_name, coords_3d, subtitle) in enumerate(methods_3d):
            ax = fig_3d.add_subplot(1, n_methods, col+1, projection='3d')
            
            if ref_idx is not None:
                seq_coords = coords_3d[1:]
            else:
                seq_coords = coords_3d
            
            # Plotear trayectoria
            ax.plot(seq_coords[:, 0], seq_coords[:, 1], seq_coords[:, 2],
                   'gray', alpha=0.3, linewidth=1)
            
            # Plotear puntos
            scatter = ax.scatter(seq_coords[:, 0], seq_coords[:, 1], seq_coords[:, 2],
                               c=frame_numbers, cmap='viridis', alpha=0.7, s=30)
            
            # Marcar inicio y fin
            ax.scatter(seq_coords[0, 0], seq_coords[0, 1], seq_coords[0, 2],
                      c='green', s=150, marker='o', label='Inicio',
                      edgecolors='black', linewidth=2)
            ax.scatter(seq_coords[-1, 0], seq_coords[-1, 1], seq_coords[-1, 2],
                      c='red', s=150, marker='s', label='Fin',
                      edgecolors='black', linewidth=2)
            
            if ref_idx is not None:
                ax.scatter(coords_3d[ref_idx, 0], coords_3d[ref_idx, 1], coords_3d[ref_idx, 2],
                          c='orange', s=250, marker='*', label='Referencia',
                          edgecolors='black', linewidth=2)
            
            ax.set_title(f'{method_name}\n{subtitle}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
            ax.set_zlabel('Dim 3')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        fig_3d.suptitle('Análisis de Secuencia 3D', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return {
            'fig_2d': fig_2d,
            'fig_3d': fig_3d,
            'pca_2d': embeddings_pca_2d,
            'tsne_2d': embeddings_tsne_2d,
            'umap_2d': embeddings_umap_2d if has_umap else None
        }
    
    def calculate_temporal_distances(self, embeddings, metadata):
        """Calcular distancias entre frames consecutivos"""
        
        print("\n" + "="*60)
        print("ANÁLISIS DE DISTANCIAS TEMPORALES")
        print("="*60)
        
        distances = []
        for i in range(len(embeddings) - 1):
            # Distancia coseno entre frames consecutivos
            dist = 1 - np.dot(embeddings[i].flatten(), embeddings[i+1].flatten()) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]))
            distances.append(dist)
        
        distances = np.array(distances)
        frame_numbers = [m['frame_number'] for m in metadata[:-1]]
        
        print(f"Distancia mínima entre frames: {distances.min():.6f}")
        print(f"Distancia máxima entre frames: {distances.max():.6f}")
        print(f"Distancia promedio: {distances.mean():.6f}")
        print(f"Desviación estándar: {distances.std():.6f}")
        
        # Gráfica de distancias
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(frame_numbers, distances, marker='o', linewidth=2, markersize=4)
        ax.axhline(distances.mean(), color='red', linestyle='--', 
                  label=f'Media: {distances.mean():.4f}')
        ax.set_xlabel('Número de Frame', fontsize=12)
        ax.set_ylabel('Distancia Coseno al Frame Siguiente', fontsize=12)
        ax.set_title('Evolución Temporal: Distancias entre Frames Consecutivos', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return distances, fig


def main():
    """Función principal"""
    
    # Configuración
    SEQUENCE_DIR = "./data/01_raw/input"
    FRAME_STEP = 5  # Extraer 1 de cada N frames
    MAX_FRAMES = None  # None = todos, o un número específico
    REFERENCE_FRAME = "./data/01_raw/input/deaging_001.png"  # Opcional, None si no hay
    
    # Crear analizador
    print("Inicializando analizador de secuencia DINOv3...")
    analyzer = SequenceAnalyzer(
        model_name='facebook/dinov3-vitb16-pretrain-lvd1689m'
    )
    
    try:
        # Analizar secuencia
        embeddings, metadata, reference_embedding = analyzer.analyze_sequence(
            SEQUENCE_DIR,
            frame_step=FRAME_STEP,
            max_frames=MAX_FRAMES,
            reference_frame=REFERENCE_FRAME
        )
        
        # Visualizar embeddings
        viz_results = analyzer.visualize_sequence(embeddings, metadata, reference_embedding)
        
        # Calcular distancias temporales
        distances, fig_dist = analyzer.calculate_temporal_distances(embeddings, metadata)
        
        # Mostrar resultados
        plt.show()
        
        # Guardar resultados
        save_results = input("\n¿Desea guardar los resultados? (s/n): ")
        if save_results.lower() == 's':
            timestamp = np.datetime64('now').astype(str).replace(':', '-')
            
            viz_results['fig_2d'].savefig(f'sequence_analysis_2d_{timestamp}.png', 
                          dpi=150, bbox_inches='tight')
            viz_results['fig_3d'].savefig(f'sequence_analysis_3d_{timestamp}.png', 
                          dpi=150, bbox_inches='tight')
            fig_dist.savefig(f'sequence_temporal_distances_{timestamp}.png',
                           dpi=150, bbox_inches='tight')
            
            # Guardar embeddings
            save_dict = {
                'embeddings': embeddings,
                'metadata': metadata,
                'distances': distances,
                'pca_2d': viz_results['pca_2d'],
                'tsne_2d': viz_results['tsne_2d'],
            }
            
            if viz_results['umap_2d'] is not None:
                save_dict['umap_2d'] = viz_results['umap_2d']
            
            if reference_embedding is not None:
                save_dict['reference_embedding'] = reference_embedding
            
            np.savez_compressed(f'sequence_embeddings_{timestamp}.npz', **save_dict)
            
            print(f"\n✅ Resultados guardados con timestamp: {timestamp}")
            print(f"   - sequence_analysis_2d_{timestamp}.png")
            print(f"   - sequence_analysis_3d_{timestamp}.png")
            print(f"   - sequence_temporal_distances_{timestamp}.png")
            print(f"   - sequence_embeddings_{timestamp}.npz")
        
    except Exception as e:
        print(f"Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
