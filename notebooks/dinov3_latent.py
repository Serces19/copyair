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
        """Agregar ruido gaussiano (mismo ruido en todos los canales para preservar color)"""
        if seed is not None:
            torch.manual_seed(seed)
        
        # Generar ruido para cada canal independientemente
        noise = torch.randn_like(image_tensor) * intensity
        noisy_image = image_tensor + noise
        return torch.clamp(noisy_image, -3, 3)  # Clamp en rango de valores normalizados
    
    @staticmethod
    def add_salt_pepper_noise(image_tensor, intensity=0.05, seed=None):
        """Agregar ruido sal y pimienta (afecta todos los canales por igual)"""
        if seed is not None:
            torch.manual_seed(seed)
        
        noisy_image = image_tensor.clone()
        
        # Crear máscara para sal y pimienta (misma para todos los canales)
        # intensity se divide entre 2 (mitad sal, mitad pimienta)
        mask_salt = torch.rand(image_tensor.shape[1:]) < (intensity / 2)
        mask_pepper = torch.rand(image_tensor.shape[1:]) < (intensity / 2)
        
        # Aplicar sal (blanco = valor alto después de normalización)
        for c in range(image_tensor.shape[0]):
            noisy_image[c][mask_salt] = 2.5  # Valor alto en espacio normalizado
            noisy_image[c][mask_pepper] = -2.5  # Valor bajo en espacio normalizado
        
        return noisy_image
    
    @staticmethod
    def add_speckle_noise(image_tensor, intensity=0.1, seed=None):
        """Agregar ruido speckle (multiplicativo)"""
        if seed is not None:
            torch.manual_seed(seed)
        
        # Ruido multiplicativo - genera ruido para cada canal
        noise = torch.randn_like(image_tensor) * intensity
        noisy_image = image_tensor * (1 + noise)
        return torch.clamp(noisy_image, -3, 3)  # Clamp en rango de valores normalizados


class TransformationGenerator:
    """Clase para generar transformaciones geométricas y de color sutiles"""
    
    @staticmethod
    def apply_rotation(image_tensor, angle_degrees=5.0, seed=None):
        """Aplicar rotación sutil"""
        if seed is not None:
            torch.manual_seed(seed)
        
        import torchvision.transforms.functional as TF
        
        # Desnormalizar temporalmente para la transformación
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = image_tensor * std + mean
        img = torch.clamp(img, 0, 1)
        
        # Aplicar rotación
        rotated = TF.rotate(img, angle=angle_degrees, interpolation=TF.InterpolationMode.BILINEAR)
        
        # Re-normalizar
        rotated = (rotated - mean) / std
        return rotated
    
    @staticmethod
    def apply_scale(image_tensor, scale_factor=1.1, seed=None):
        """Aplicar zoom/escala sutil"""
        if seed is not None:
            torch.manual_seed(seed)
        
        import torchvision.transforms.functional as TF
        
        # Desnormalizar
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = image_tensor * std + mean
        img = torch.clamp(img, 0, 1)
        
        # Calcular nuevo tamaño
        _, h, w = img.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        # Resize y crop al centro
        scaled = TF.resize(img, [new_h, new_w], interpolation=TF.InterpolationMode.BILINEAR)
        scaled = TF.center_crop(scaled, [h, w])
        
        # Re-normalizar
        scaled = (scaled - mean) / std
        return scaled
    
    @staticmethod
    def apply_brightness(image_tensor, brightness_factor=1.2, seed=None):
        """Aplicar cambio de brillo sutil"""
        if seed is not None:
            torch.manual_seed(seed)
        
        import torchvision.transforms.functional as TF
        
        # Desnormalizar
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = image_tensor * std + mean
        img = torch.clamp(img, 0, 1)
        
        # Aplicar brillo
        bright = TF.adjust_brightness(img, brightness_factor)
        bright = torch.clamp(bright, 0, 1)
        
        # Re-normalizar
        bright = (bright - mean) / std
        return bright
    
    @staticmethod
    def apply_contrast(image_tensor, contrast_factor=1.2, seed=None):
        """Aplicar cambio de contraste sutil"""
        if seed is not None:
            torch.manual_seed(seed)
        
        import torchvision.transforms.functional as TF
        
        # Desnormalizar
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = image_tensor * std + mean
        img = torch.clamp(img, 0, 1)
        
        # Aplicar contraste
        contrast = TF.adjust_contrast(img, contrast_factor)
        contrast = torch.clamp(contrast, 0, 1)
        
        # Re-normalizar
        contrast = (contrast - mean) / std
        return contrast
    
    @staticmethod
    def apply_perspective(image_tensor, distortion_scale=0.2, seed=None):
        """Aplicar distorsión de perspectiva sutil"""
        if seed is not None:
            torch.manual_seed(seed)
        
        import torchvision.transforms.functional as TF
        
        # Desnormalizar
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = image_tensor * std + mean
        img = torch.clamp(img, 0, 1)
        
        # Obtener dimensiones
        _, h, w = img.shape
        
        # Definir puntos de perspectiva con distorsión sutil
        startpoints = [[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]
        
        # Aplicar distorsión aleatoria a los puntos
        torch.manual_seed(seed if seed else 42)
        distortion = distortion_scale * min(h, w)
        endpoints = [
            [int(startpoints[i][0] + torch.randn(1).item() * distortion),
             int(startpoints[i][1] + torch.randn(1).item() * distortion)]
            for i in range(4)
        ]
        
        # Aplicar transformación de perspectiva
        perspective = TF.perspective(img, startpoints, endpoints, 
                                     interpolation=TF.InterpolationMode.BILINEAR)
        
        # Re-normalizar
        perspective = (perspective - mean) / std
        return perspective


class DINOv3RobustnessAnalyzer:
    """Analizador de robustez de embeddings DINOv3 frente al ruido"""
    
    def __init__(self, model_name='facebook/dinov3-vitb16-pretrain-lvd1689m'):
        self.embedder = DINOv3Embedder(model_name=model_name, pooling='mean')
        self.noise_generator = NoiseGenerator()
        self.transformation_generator = TransformationGenerator()
        
    def analyze_image_robustness(self, image_path, n_trials=30, 
                                  noise_intensities=[0.1],
                                  transform_intensities=None):
        """Analizar robustez de embeddings para una imagen con ruido y transformaciones
        
        Args:
            image_path: Ruta a la imagen
            n_trials: Número de trials por cada intensidad
            noise_intensities: Lista de intensidades para ruido Gaussiano
            transform_intensities: Dict con intensidades para transformaciones, ej:
                {
                    'rotation': [2, 5, 10],  # grados
                    'scale': [1.05, 1.1, 1.15],  # factor
                    'brightness': [0.9, 1.1, 1.2],  # factor
                    'contrast': [0.9, 1.1, 1.2],  # factor
                    'perspective': [0.05, 0.1, 0.15]  # escala de distorsión
                }
        """
        
        print(f"\n{'='*60}")
        print(f"ANALIZANDO IMAGEN: {image_path}")
        print(f"{'='*60}")
        
        # Cargar y procesar imagen original
        original_tensor, original_image = self.embedder.process_image(image_path)
        original_embedding = self.embedder.get_embedding(original_tensor)
        
        print(f"Dimensión del embedding original: {original_embedding.shape}")
        
        # Generar embeddings con ruido y transformaciones
        all_embeddings = [original_embedding]
        metadata = [{'type': 'original', 'category': 'original', 'intensity': 0, 'seed': None}]
        
        # 1. RUIDO GAUSSIANO
        print(f"\n[1/2] Generando embeddings con ruido Gaussiano...")
        for trial in range(n_trials):
            for intensity in noise_intensities:
                seed = trial * 100 + hash('gaussian') % 1000
                
                # Aplicar ruido
                noisy_tensor = self.noise_generator.add_gaussian_noise(
                    original_tensor.clone(), intensity=intensity, seed=seed
                )
                
                # Obtener embedding
                noisy_embedding = self.embedder.get_embedding(noisy_tensor)
                
                all_embeddings.append(noisy_embedding)
                metadata.append({
                    'type': 'gaussian_noise',
                    'category': 'noise',
                    'intensity': intensity,
                    'seed': seed,
                    'trial': trial
                })
        
        # 2. TRANSFORMACIONES GEOMÉTRICAS Y DE COLOR
        if transform_intensities is None:
            transform_intensities = {
                'rotation': [1, 3],
                'scale': [1.05, 1.1],
                'brightness': [0.98, 1.02],
                'contrast': [0.95, 1.05],
                'perspective': [0.05, 0.1]
            }
        
        print(f"[2/2] Generando embeddings con transformaciones geométricas...")
        
        transformations = [
            ('rotation', self.transformation_generator.apply_rotation, 'angle_degrees'),
            ('scale', self.transformation_generator.apply_scale, 'scale_factor'),
            ('brightness', self.transformation_generator.apply_brightness, 'brightness_factor'),
            ('contrast', self.transformation_generator.apply_contrast, 'contrast_factor'),
            ('perspective', self.transformation_generator.apply_perspective, 'distortion_scale')
        ]
        
        for transform_name, transform_func, param_name in transformations:
            if transform_name not in transform_intensities:
                continue
                
            for trial in range(n_trials):
                for intensity in transform_intensities[transform_name]:
                    seed = trial * 100 + hash(transform_name) % 1000
                    
                    # Aplicar transformación
                    kwargs = {param_name: intensity, 'seed': seed}
                    transformed_tensor = transform_func(original_tensor.clone(), **kwargs)
                    
                    # Obtener embedding
                    transformed_embedding = self.embedder.get_embedding(transformed_tensor)
                    
                    all_embeddings.append(transformed_embedding)
                    metadata.append({
                        'type': transform_name,
                        'category': 'transformation',
                        'intensity': intensity,
                        'seed': seed,
                        'trial': trial
                    })
        
        # Convertir a array numpy
        all_embeddings = np.vstack(all_embeddings)
        
        print(f"\nTotal de embeddings generados: {all_embeddings.shape[0]}")
        print(f"  → Ruido Gaussiano: {len([m for m in metadata if m['category'] == 'noise'])}")
        print(f"  → Transformaciones: {len([m for m in metadata if m['category'] == 'transformation'])}")
        
        return all_embeddings, metadata, original_tensor, original_embedding
    
    def analyze_image_sequence(self, sequence_dir, frame_step=5, max_frames=None):
        """Extraer embeddings de una secuencia de imágenes
        
        Args:
            sequence_dir: Directorio con la secuencia de imágenes
            frame_step: Extraer 1 de cada N frames (default: 5)
            max_frames: Máximo número de frames a procesar (None = todos)
        
        Returns:
            embeddings: Array de embeddings
            metadata: Lista de metadatos por frame
            frame_paths: Lista de rutas a los frames procesados
        """
        import os
        from pathlib import Path
        
        print(f"\n{'='*60}")
        print(f"ANALIZANDO SECUENCIA: {sequence_dir}")
        print(f"{'='*60}")
        
        # Buscar todas las imágenes en el directorio
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
        
        # Extraer embeddings
        embeddings = []
        metadata = []
        frame_paths = []
        
        for idx, frame_path in enumerate(selected_frames):
            # Extraer número de frame del nombre del archivo
            frame_num = int(''.join(filter(str.isdigit, frame_path.stem)))
            
            # Procesar imagen
            tensor, _ = self.embedder.process_image(str(frame_path))
            embedding = self.embedder.get_embedding(tensor)
            
            embeddings.append(embedding)
            metadata.append({
                'type': 'sequence_frame',
                'category': 'sequence',
                'frame_number': frame_num,
                'frame_index': idx,
                'path': str(frame_path)
            })
            frame_paths.append(str(frame_path))
            
            if (idx + 1) % 10 == 0:
                print(f"  Procesados {idx + 1}/{len(selected_frames)} frames...")
        
        embeddings = np.vstack(embeddings)
        
        print(f"\n✓ Embeddings de secuencia extraídos: {embeddings.shape}")
        
        return embeddings, metadata, frame_paths
    
    def visualize_embeddings(self, embeddings, metadata, original_idx=0):
        """Visualizar embeddings usando múltiples métodos de reducción dimensional"""
        
        print("\n" + "="*60)
        print("REDUCCIÓN DIMENSIONAL Y VISUALIZACIÓN")
        print("="*60)
        
        # Preparar colores y etiquetas - ahora distinguimos por categoría
        categories = [m['category'] for m in metadata]
        types = [m['type'] for m in metadata]
        intensities = np.array([m.get('intensity', 0) for m in metadata])  # Usar .get() para frames de secuencia
        
        # Colores por tipo
        type_colors = {
            'original': 'green',
            'gaussian_noise': 'red',
            'rotation': 'blue',
            'scale': 'purple',
            'brightness': 'orange',
            'contrast': 'cyan',
            'perspective': 'magenta',
            'sequence_frame': 'darkviolet'  # Color para frames de secuencia
        }
        
        # Colores por categoría (para visualización simplificada)
        category_colors = {
            'original': 'green',
            'noise': 'red',
            'transformation': 'blue',
            'sequence': 'darkviolet'  # Color para secuencia
        }
        
        # ============================================================
        # MÉTODO 1: PCA (Análisis de Componentes Principales)
        # ============================================================
        print("\n[1/3] Aplicando PCA...")
        print("  → PCA preserva la varianza máxima de los datos")
        print("  → Es un método lineal y determinístico")
        
        pca_2d = PCA(n_components=2, random_state=42)
        embeddings_pca_2d = pca_2d.fit_transform(embeddings)
        variance_2d = pca_2d.explained_variance_ratio_
        print(f"  → Varianza explicada: PC1={variance_2d[0]:.2%}, PC2={variance_2d[1]:.2%}, Total={variance_2d.sum():.2%}")
        
        pca_3d = PCA(n_components=3, random_state=42)
        embeddings_pca_3d = pca_3d.fit_transform(embeddings)
        variance_3d = pca_3d.explained_variance_ratio_
        print(f"  → Varianza 3D: PC1={variance_3d[0]:.2%}, PC2={variance_3d[1]:.2%}, PC3={variance_3d[2]:.2%}, Total={variance_3d.sum():.2%}")
        
        # ============================================================
        # MÉTODO 2: t-SNE (t-Distributed Stochastic Neighbor Embedding)
        # ============================================================
        print("\n[2/3] Aplicando t-SNE...")
        print("  → t-SNE preserva la estructura local (vecinos cercanos)")
        print("  → Es no-lineal y puede distorsionar distancias globales")
        
        # Usar PCA para pre-reducir dimensionalidad (acelera t-SNE)
        pca_50 = PCA(n_components=min(50, embeddings.shape[0]-1), random_state=42)
        embeddings_pca_50 = pca_50.fit_transform(embeddings)
        
        perplexity = min(30, embeddings.shape[0] // 3)
        tsne_2d = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        embeddings_tsne_2d = tsne_2d.fit_transform(embeddings_pca_50)
        
        tsne_3d = TSNE(n_components=3, perplexity=perplexity, random_state=42, n_iter=1000)
        embeddings_tsne_3d = tsne_3d.fit_transform(embeddings_pca_50)
        print(f"  → Perplexity usada: {perplexity}")
        
        # ============================================================
        # MÉTODO 3: UMAP (Uniform Manifold Approximation and Projection)
        # ============================================================
        print("\n[3/3] Aplicando UMAP...")
        print("  → UMAP preserva tanto estructura local como global")
        print("  → Generalmente más rápido que t-SNE y preserva mejor las distancias")
        
        try:
            from umap import UMAP
            
            n_neighbors = min(15, embeddings.shape[0] // 2)
            umap_2d = UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42, min_dist=0.1)
            embeddings_umap_2d = umap_2d.fit_transform(embeddings)
            
            umap_3d = UMAP(n_components=3, n_neighbors=n_neighbors, random_state=42, min_dist=0.1)
            embeddings_umap_3d = umap_3d.fit_transform(embeddings)
            print(f"  → n_neighbors usada: {n_neighbors}")
            has_umap = True
        except ImportError:
            print("  ⚠ UMAP no disponible (instalar con: pip install umap-learn)")
            has_umap = False
        
        # ============================================================
        # VISUALIZACIONES 2D
        # ============================================================
        n_methods = 3 if has_umap else 2
        fig_2d, axes = plt.subplots(3, n_methods, figsize=(6*n_methods, 18))
        
        methods_2d = [
            ('PCA', embeddings_pca_2d, f'Varianza explicada: {variance_2d.sum():.1%}'),
            ('t-SNE', embeddings_tsne_2d, f'Perplexity: {perplexity}'),
        ]
        if has_umap:
            methods_2d.append(('UMAP', embeddings_umap_2d, f'n_neighbors: {n_neighbors}'))
        
        for col, (method_name, coords_2d, subtitle) in enumerate(methods_2d):
            # Fila 1: Por categoría (original, ruido, transformación, secuencia)
            ax = axes[0, col]
            for category in ['noise', 'transformation', 'sequence', 'original']:
                mask = np.array([c == category for c in categories])
                if mask.sum() > 0:
                    label = category.capitalize() if category != 'original' else 'Original'
                    ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1], 
                              c=category_colors[category], label=label, alpha=0.6, s=30)
            
            ax.scatter(coords_2d[original_idx, 0], coords_2d[original_idx, 1], 
                      c='black', s=300, marker='*', label='Original (marcado)', 
                      edgecolors='yellow', linewidth=2, zorder=10)
            ax.set_title(f'{method_name} - Por categoría\n{subtitle}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Dimensión 1')
            ax.set_ylabel('Dimensión 2')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Fila 2: Por tipo específico (gaussian_noise, rotation, scale, etc.)
            ax = axes[1, col]
            for type_name in type_colors.keys():
                mask = np.array([t == type_name for t in types])
                if mask.sum() > 0:
                    # Formatear nombre para la leyenda
                    label = type_name.replace('_', ' ').title()
                    ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1], 
                              c=type_colors[type_name], label=label, alpha=0.6, s=30)
            
            ax.scatter(coords_2d[original_idx, 0], coords_2d[original_idx, 1], 
                      c='black', s=300, marker='*', 
                      edgecolors='yellow', linewidth=2, zorder=10)
            ax.set_title(f'{method_name} - Por tipo específico\n{subtitle}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Dimensión 1')
            ax.set_ylabel('Dimensión 2')
            ax.legend(loc='best', fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)
            
            # Fila 3: Por intensidad
            ax = axes[2, col]
            scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                               c=intensities, cmap='plasma', alpha=0.6, s=30)
            ax.scatter(coords_2d[original_idx, 0], coords_2d[original_idx, 1], 
                      c='black', s=300, marker='*', 
                      edgecolors='yellow', linewidth=2, zorder=10)
            ax.set_title(f'{method_name} - Por intensidad\n{subtitle}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Dimensión 1')
            ax.set_ylabel('Dimensión 2')
            plt.colorbar(scatter, ax=ax, label='Intensidad')
            ax.grid(True, alpha=0.3)
        
        fig_2d.suptitle('Comparación de Métodos de Reducción Dimensional (2D)', 
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
            
            # Plotear por categoría (incluyendo secuencia)
            for category in ['noise', 'transformation', 'sequence', 'original']:
                mask = np.array([c == category for c in categories])
                if mask.sum() > 0:
                    label = category.capitalize() if category != 'original' else 'Original'
                    ax.scatter(coords_3d[mask, 0], coords_3d[mask, 1], coords_3d[mask, 2],
                              c=category_colors[category], label=label, alpha=0.6, s=20)
            
            # Marcar original
            ax.scatter(coords_3d[original_idx, 0], coords_3d[original_idx, 1], coords_3d[original_idx, 2],
                      c='black', s=300, marker='*', 
                      edgecolors='yellow', linewidth=2, zorder=10)
            
            ax.set_title(f'{method_name}\n{subtitle}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
            ax.set_zlabel('Dim 3')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        fig_3d.suptitle('Visualización 3D de Embeddings', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # ============================================================
        # ANÁLISIS DE DISTORSIÓN
        # ============================================================
        print("\n" + "="*60)
        print("ANÁLISIS DE DISTORSIÓN DE DISTANCIAS")
        print("="*60)
        
        # Calcular distancias reales en espacio de embeddings
        from scipy.spatial.distance import pdist, squareform
        real_distances = squareform(pdist(embeddings, metric='cosine'))
        
        # Calcular distancias en espacios reducidos
        pca_distances = squareform(pdist(embeddings_pca_2d, metric='euclidean'))
        tsne_distances = squareform(pdist(embeddings_tsne_2d, metric='euclidean'))
        
        # Normalizar para comparar
        pca_distances = pca_distances / pca_distances.max()
        tsne_distances = tsne_distances / tsne_distances.max()
        real_distances_norm = real_distances / real_distances.max()
        
        # Correlación entre distancias reales y reducidas
        from scipy.stats import spearmanr
        corr_pca = spearmanr(real_distances.flatten(), pca_distances.flatten())[0]
        corr_tsne = spearmanr(real_distances.flatten(), tsne_distances.flatten())[0]
        
        print(f"Correlación Spearman (distancias reales vs reducidas):")
        print(f"  → PCA:   {corr_pca:.4f}")
        print(f"  → t-SNE: {corr_tsne:.4f}")
        if has_umap:
            umap_distances = squareform(pdist(embeddings_umap_2d, metric='euclidean'))
            umap_distances = umap_distances / umap_distances.max()
            corr_umap = spearmanr(real_distances.flatten(), umap_distances.flatten())[0]
            print(f"  → UMAP:  {corr_umap:.4f}")
        
        print("\n💡 Interpretación:")
        print("  - Correlación cercana a 1.0 = método preserva bien las distancias")
        print("  - PCA preserva distancias globales pero es lineal")
        print("  - t-SNE preserva vecindarios locales, puede distorsionar distancias globales")
        if has_umap:
            print("  - UMAP balancea estructura local y global")
        
        return {
            'fig_2d': fig_2d,
            'fig_3d': fig_3d,
            'pca_2d': embeddings_pca_2d,
            'tsne_2d': embeddings_tsne_2d,
            'umap_2d': embeddings_umap_2d if has_umap else None,
            'pca_3d': embeddings_pca_3d,
            'tsne_3d': embeddings_tsne_3d,
            'umap_3d': embeddings_umap_3d if has_umap else None,
        }
    
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
    
    def visualize_noisy_images(self, original_tensor, noise_intensities=[0.01, 0.05, 0.1, 0.2]):
        """Visualizar la imagen original con diferentes niveles de ruido Gaussiano"""
        
        print("\n" + "="*60)
        print("VISUALIZACIÓN DE RUIDO GAUSSIANO APLICADO")
        print("="*60)
        
        # Convertir tensor a imagen para visualización
        def tensor_to_img(tensor):
            """Convertir tensor normalizado a imagen numpy [0, 255]"""
            # Desnormalizar
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = tensor * std + mean
            img = torch.clamp(img, 0, 1)
            # Convertir a numpy y transponer
            img = img.permute(1, 2, 0).numpy()
            return (img * 255).astype(np.uint8)
        
        # Crear figura: 1 fila con original + intensidades
        n_cols = len(noise_intensities) + 1
        
        fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 4))
        
        # Primera columna: Imagen original
        original_img = tensor_to_img(original_tensor)
        axes[0].imshow(original_img)
        axes[0].set_title('ORIGINAL\n(Sin ruido)', fontsize=11, fontweight='bold')
        axes[0].axis('off')
        
        # Resto de columnas: Diferentes intensidades de ruido Gaussiano
        for col, intensity in enumerate(noise_intensities, start=1):
            # Aplicar ruido Gaussiano
            noisy_tensor = self.noise_generator.add_gaussian_noise(
                original_tensor.clone(), intensity=intensity, seed=42
            )
            noisy_img = tensor_to_img(noisy_tensor)
            
            # Mostrar imagen
            axes[col].imshow(noisy_img)
            axes[col].set_title(f'Ruido Gaussiano\nIntensidad: {intensity}', 
                              fontsize=11, fontweight='bold')
            axes[col].axis('off')
        
        fig.suptitle('Comparación Visual: Imagen Original vs. Diferentes Niveles de Ruido Gaussiano', 
                    fontsize=13, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        print("✓ Visualización de ruido Gaussiano generada")
        print(f"  → Intensidades: {noise_intensities}")
        
        return fig


def main():
    """Función principal"""
    
    # Configuración
    IMAGE_PATH = "./data/01_raw/input/deaging_001.png"  # Cambiar por la ruta de tu imagen
    N_TRIALS = 15  # Número de intentos por tipo/intensidad
    NOISE_INTENSITIES = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    
    # Configurar transformaciones geométricas (valores sutiles)
    TRANSFORM_INTENSITIES = {
        'rotation': [1, 2, 3, 5],  # grados de rotación
        'scale': [1.02, 1.05, 1.08, 1.1],  # factor de escala
        'brightness': [0.95, 0.98, 1.02, 1.05],  # factor de brillo
        'contrast': [0.95, 0.98, 1.02, 1.05],  # factor de contraste
        'perspective': [0.02, 0.05, 0.08, 0.1]  # escala de distorsión
    }
    
    # Crear analizador
    print("Inicializando analizador DINOv3...")
    analyzer = DINOv3RobustnessAnalyzer(
        model_name='facebook/dinov3-vitb16-pretrain-lvd1689m'
    )
    
    # ============================================================
    # TEST DE DETERMINISMO
    # ============================================================
    print("\n" + "="*60)
    print("TEST DE DETERMINISMO")
    print("="*60)
    print("Verificando que la misma imagen produce el mismo embedding...")
    
    try:
        tensor1, _ = analyzer.embedder.process_image(IMAGE_PATH)
        embedding1 = analyzer.embedder.get_embedding(tensor1)
        
        tensor2, _ = analyzer.embedder.process_image(IMAGE_PATH)
        embedding2 = analyzer.embedder.get_embedding(tensor2)
        
        # Calcular diferencia
        diff = np.abs(embedding1 - embedding2).max()
        cosine_sim = np.dot(embedding1.flatten(), embedding2.flatten()) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        
        print(f"✓ Diferencia máxima entre embeddings: {diff:.10f}")
        print(f"✓ Similitud coseno: {cosine_sim:.10f}")
        
        if diff < 1e-6:
            print("✅ DETERMINISMO CONFIRMADO: Misma imagen → Mismo embedding")
        else:
            print("⚠️  ADVERTENCIA: Los embeddings difieren ligeramente")
    except Exception as e:
        print(f"❌ Error en test de determinismo: {e}")
    
    # ============================================================
    # ANÁLISIS DE ROBUSTEZ (RUIDO + TRANSFORMACIONES)
    # ============================================================
    try:
        embeddings, metadata, original_tensor, original_embedding = \
            analyzer.analyze_image_robustness(
                IMAGE_PATH, 
                n_trials=N_TRIALS,
                noise_intensities=NOISE_INTENSITIES,
                transform_intensities=TRANSFORM_INTENSITIES
            )
        
        # Visualizar cómo se ve la imagen con diferentes ruidos
        fig_noisy = analyzer.visualize_noisy_images(original_tensor, NOISE_INTENSITIES)
        
        # Visualizar embeddings (retorna diccionario ahora)
        viz_results = analyzer.visualize_embeddings(embeddings, metadata)
        
        # Calcular distancias
        distances, fig_dist = analyzer.calculate_distances(embeddings)
        
        # Mostrar resultados
        plt.show()
        
        # Guardar resultados si se desea
        save_results = input("\n¿Desea guardar los resultados? (s/n): ")
        if save_results.lower() == 's':
            timestamp = np.datetime64('now').astype(str).replace(':', '-')
            
            # Guardar figuras 2D y 3D
            viz_results['fig_2d'].savefig(f'dinov3_robustness_2d_{timestamp}.png', 
                          dpi=150, bbox_inches='tight')
            viz_results['fig_3d'].savefig(f'dinov3_robustness_3d_{timestamp}.png', 
                          dpi=150, bbox_inches='tight')
            fig_dist.savefig(f'dinov3_distance_distribution_{timestamp}.png', 
                           dpi=150, bbox_inches='tight')
            fig_noisy.savefig(f'dinov3_noisy_images_{timestamp}.png',
                            dpi=150, bbox_inches='tight')
            
            # Guardar embeddings y coordenadas reducidas
            save_dict = {
                'embeddings': embeddings,
                'distances': distances,
                'metadata': metadata,
                'pca_2d': viz_results['pca_2d'],
                'tsne_2d': viz_results['tsne_2d'],
                'pca_3d': viz_results['pca_3d'],
                'tsne_3d': viz_results['tsne_3d'],
            }
            
            if viz_results['umap_2d'] is not None:
                save_dict['umap_2d'] = viz_results['umap_2d']
                save_dict['umap_3d'] = viz_results['umap_3d']
            
            np.savez_compressed(f'dinov3_embeddings_{timestamp}.npz', **save_dict)
            
            print(f"\n✅ Resultados guardados con timestamp: {timestamp}")
            print(f"   - dinov3_robustness_2d_{timestamp}.png")
            print(f"   - dinov3_robustness_3d_{timestamp}.png")
            print(f"   - dinov3_distance_distribution_{timestamp}.png")
            print(f"   - dinov3_noisy_images_{timestamp}.png")
            print(f"   - dinov3_embeddings_{timestamp}.npz")
        
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