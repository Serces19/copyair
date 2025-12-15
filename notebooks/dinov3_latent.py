import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from torchvision import transforms
from PIL import Image
import random
import warnings
warnings.filterwarnings('ignore')

class DINOv3Embedder:
    """Clase para extraer embeddings usando DINOv3"""
    
    CANDIDATES = ['facebook/dinov2-base', 'facebook/dinov3-vitb16-pretrain-lvd1689m']
    
    def __init__(self,
                 model_name=None,
                 device='cuda',
                 input_size=224,
                 pooling='mean',    # 'tokens', 'mean', 'cls'
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
                # Usar el token [CLS]
                features = outputs.last_hidden_state[:, 0, :]
            elif self.pooling == 'mean':
                # Promedio de todos los tokens
                features = outputs.last_hidden_state.mean(dim=1)
            elif self.pooling == 'tokens':
                # Todos los tokens (aplanados)
                features = outputs.last_hidden_state.flatten(1)
            
            # Normalizar embeddings
            features = features / features.norm(dim=-1, keepdim=True)
            
            return features.cpu().numpy()
    
    def process_image(self, image_path):
        """Procesar imagen desde ruta"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)
        return image_tensor, image


class NoiseGenerator:
    """Clase para generar diferentes tipos de ruido en imágenes"""
    
    @staticmethod
    def add_gaussian_noise(image_tensor, intensity=0.1, seed=None):
        """Agregar ruido gaussiano"""
        if seed is not None:
            torch.manual_seed(seed)
        
        noise = torch.randn_like(image_tensor) * intensity
        noisy_image = image_tensor + noise
        return torch.clamp(noisy_image, 0, 1)
    
    @staticmethod
    def add_salt_pepper_noise(image_tensor, intensity=0.05, seed=None):
        """Agregar ruido sal y pimienta"""
        if seed is not None:
            torch.manual_seed(seed)
        
        noisy_image = image_tensor.clone()
        mask = torch.rand_like(image_tensor[0]) < intensity
        noisy_image[0][mask] = torch.randint(0, 2, (mask.sum(),)).float()
        return noisy_image
    
    @staticmethod
    def add_speckle_noise(image_tensor, intensity=0.1, seed=None):
        """Agregar ruido speckle"""
        if seed is not None:
            torch.manual_seed(seed)
        
        noise = torch.randn_like(image_tensor) * intensity
        noisy_image = image_tensor * (1 + noise)
        return torch.clamp(noisy_image, 0, 1)


class DINOv3RobustnessAnalyzer:
    """Analizador de robustez de embeddings DINOv3 frente al ruido"""
    
    def __init__(self, model_name='facebook/dinov3-vitb16-pretrain-lvd1689m'):
        self.embedder = DINOv3Embedder(model_name=model_name, pooling='mean')
        self.noise_generator = NoiseGenerator()
        
    def analyze_image_robustness(self, image_path, n_trials=30, noise_intensities=[0.01, 0.05, 0.1, 0.2]):
        """Analizar robustez de embeddings para una imagen"""
        
        print(f"\n{'='*60}")
        print(f"ANALIZANDO IMAGEN: {image_path}")
        print(f"{'='*60}")
        
        # Cargar y procesar imagen original
        original_tensor, original_image = self.embedder.process_image(image_path)
        original_embedding = self.embedder.get_embedding(original_tensor)
        
        print(f"Dimensión del embedding original: {original_embedding.shape}")
        
        # Generar embeddings con ruido
        all_embeddings = [original_embedding]
        metadata = [{'type': 'original', 'intensity': 0, 'seed': None}]
        
        for trial in range(n_trials):
            for intensity in noise_intensities:
                for noise_type, noise_func in [
                    ('gaussian', self.noise_generator.add_gaussian_noise),
                    ('salt_pepper', self.noise_generator.add_salt_pepper_noise),
                    ('speckle', self.noise_generator.add_speckle_noise)
                ]:
                    seed = trial * 100 + hash(noise_type) % 1000
                    
                    # Aplicar ruido
                    noisy_tensor = noise_func(original_tensor.clone(), intensity=intensity, seed=seed)
                    
                    # Obtener embedding
                    noisy_embedding = self.embedder.get_embedding(noisy_tensor)
                    
                    all_embeddings.append(noisy_embedding)
                    metadata.append({
                        'type': noise_type,
                        'intensity': intensity,
                        'seed': seed,
                        'trial': trial
                    })
        
        # Convertir a array numpy
        all_embeddings = np.vstack(all_embeddings)
        
        print(f"Total de embeddings generados: {all_embeddings.shape[0]}")
        
        return all_embeddings, metadata, original_tensor, original_embedding
    
    def visualize_embeddings(self, embeddings, metadata, original_idx=0):
        """Visualizar embeddings en 2D usando t-SNE y PCA"""
        
        print("\nReduciendo dimensionalidad...")
        
        # Aplicar PCA primero para acelerar t-SNE
        pca = PCA(n_components=50, random_state=42)
        embeddings_pca = pca.fit_transform(embeddings)
        
        # Aplicar t-SNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
        embeddings_2d = tsne.fit_transform(embeddings_pca)
        
        # Crear figura
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Visualización por tipo de ruido
        colors = {'original': 'green', 'gaussian': 'red', 'salt_pepper': 'blue', 'speckle': 'purple'}
        noise_types = [m['type'] for m in metadata]
        
        for noise_type in colors.keys():
            mask = [nt == noise_type for nt in noise_types]
            axes[0].scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                          c=colors[noise_type], label=noise_type, alpha=0.6)
        
        axes[0].scatter(embeddings_2d[original_idx, 0], embeddings_2d[original_idx, 1], 
                       c='black', s=200, marker='*', label='Original', edgecolors='white', linewidth=2)
        axes[0].set_title('Embeddings por tipo de ruido', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Visualización por intensidad de ruido
        intensities = [m['intensity'] for m in metadata]
        scatter = axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                 c=intensities, cmap='viridis', alpha=0.6)
        axes[1].scatter(embeddings_2d[original_idx, 0], embeddings_2d[original_idx, 1], 
                       c='black', s=200, marker='*', edgecolors='white', linewidth=2)
        axes[1].set_title('Embeddings por intensidad de ruido', fontsize=14)
        plt.colorbar(scatter, ax=axes[1], label='Intensidad')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Visualización con PCA (2 componentes)
        pca_2d = PCA(n_components=2, random_state=42)
        embeddings_pca_2d = pca_2d.fit_transform(embeddings)
        
        axes[2].scatter(embeddings_pca_2d[1:, 0], embeddings_pca_2d[1:, 1], 
                       c='red', alpha=0.6, label='Con ruido')
        axes[2].scatter(embeddings_pca_2d[original_idx, 0], embeddings_pca_2d[original_idx, 1], 
                       c='green', s=200, marker='*', label='Original', edgecolors='black', linewidth=2)
        axes[2].set_title('Visualización PCA (2 componentes)', fontsize=14)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle('Análisis de Robustez de Embeddings DINOv3', fontsize=16)
        plt.tight_layout()
        
        return fig, embeddings_2d
    
    def calculate_distances(self, embeddings, original_idx=0):
        """Calcular distancias al embedding original"""
        
        original_embedding = embeddings[original_idx]
        distances = []
        
        for i, embedding in enumerate(embeddings):
            if i == original_idx:
                continue
            
            # Distancia coseno (1 - similitud coseno)
            dist = 1 - np.dot(original_embedding.flatten(), embedding.flatten()) / (
                np.linalg.norm(original_embedding) * np.linalg.norm(embedding))
            distances.append(dist)
        
        distances = np.array(distances)
        
        print("\n" + "="*60)
        print("ANÁLISIS DE DISTANCIAS:")
        print("="*60)
        print(f"Distancia mínima al original: {distances.min():.6f}")
        print(f"Distancia máxima al original: {distances.max():.6f}")
        print(f"Distancia promedio: {distances.mean():.6f}")
        print(f"Desviación estándar: {distances.std():.6f}")
        
        # Histograma de distancias
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(distances, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(distances.mean(), color='red', linestyle='--', 
                  label=f'Media: {distances.mean():.4f}')
        ax.set_xlabel('Distancia Coseno al Embedding Original', fontsize=12)
        ax.set_ylabel('Frecuencia', fontsize=12)
        ax.set_title('Distribución de Distancias de Embeddings con Ruido', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return distances, fig


def main():
    """Función principal"""
    
    # Configuración
    IMAGE_PATH = "./data/01_raw/input/deaging_001.png"  # Cambiar por la ruta de tu imagen
    N_TRIALS = 10  # Número de intentos por tipo/intensidad de ruido
    NOISE_INTENSITIES = [0.01, 0.05, 0.1, 0.15, 0.2]
    
    # Crear analizador
    print("Inicializando analizador DINOv3...")
    analyzer = DINOv3RobustnessAnalyzer(
        model_name='facebook/dinov3-vitb16-pretrain-lvd1689m'
    )
    
    # Analizar robustez
    try:
        embeddings, metadata, original_tensor, original_embedding = \
            analyzer.analyze_image_robustness(
                IMAGE_PATH, 
                n_trials=N_TRIALS,
                noise_intensities=NOISE_INTENSITIES
            )
        
        # Visualizar embeddings
        fig_viz, embeddings_2d = analyzer.visualize_embeddings(embeddings, metadata)
        
        # Calcular distancias
        distances, fig_dist = analyzer.calculate_distances(embeddings)
        
        # Mostrar resultados
        plt.show()
        
        # Guardar resultados si se desea
        save_results = input("\n¿Desea guardar los resultados? (s/n): ")
        if save_results.lower() == 's':
            timestamp = np.datetime64('now').astype(str).replace(':', '-')
            
            fig_viz.savefig(f'dinov3_robustness_visualization_{timestamp}.png', 
                          dpi=150, bbox_inches='tight')
            fig_dist.savefig(f'dinov3_distance_distribution_{timestamp}.png', 
                           dpi=150, bbox_inches='tight')
            
            # Guardar embeddings
            np.savez_compressed(
                f'dinov3_embeddings_{timestamp}.npz',
                embeddings=embeddings,
                embeddings_2d=embeddings_2d,
                distances=distances,
                metadata=metadata
            )
            
            print(f"Resultados guardados con timestamp: {timestamp}")
        
    except FileNotFoundError:
        print(f"Error: No se encontró la imagen en {IMAGE_PATH}")
        print("Por favor, asegúrate de que la ruta sea correcta.")
    except Exception as e:
        print(f"Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Verificar dependencias
    # required_packages = ['torch', 'transformers', 'PIL', 'matplotlib', 'scikit-learn', 'numpy']
    
    # missing_packages = []
    # for package in required_packages:
    #     try:
    #         __import__(package)
    #     except ImportError:
    #         missing_packages.append(package)
    
    # if missing_packages:
    #     print(f"Paquetes faltantes: {missing_packages}")
    #     print("Instala con: pip install", " ".join(missing_packages))
    # else:
    #     main()

    main()