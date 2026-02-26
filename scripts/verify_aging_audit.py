import os
import torch
import numpy as np
import glob
import cv2
from PIL import Image
from skimage.exposure import match_histograms
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoImageProcessor, AutoModel

def get_cls_embedding_from_numpy(img_np, processor, model, device, input_size=None):
    image = Image.fromarray(img_np.astype('uint8'))
    
    if input_size:
        inputs = processor(images=image, size={"height": input_size, "width": input_size}, return_tensors="pt").to(device)
    else:
        inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = cls_embedding / cls_embedding.norm(dim=-1, keepdim=True)
        
    return cls_embedding.cpu().numpy().flatten()

def load_cv2_image(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def main():
    # CONFIGURATION
    INPUT_SIZE = 512
    MODEL_NAME = 'facebook/dinov3-vitb16-pretrain-lvd1689m'
    DATA_DIR = r'k:\Trabajos\Code\copyair\data\03_processed\test_age'
    REFERENCE_AGE = 18
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Loading model: {MODEL_NAME}")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    
    # Dynamically find all PNG files and extract ages
    png_files = glob.glob(os.path.join(DATA_DIR, "*.png"))
    age_files = {}
    for f in png_files:
        try:
            age = int(os.path.basename(f).split('.')[0])
            age_files[age] = f
        except ValueError:
            continue
            
    if not age_files:
        print(f"No valid aging images found in {DATA_DIR}")
        return

    ages = sorted(age_files.keys())
    print(f"Found ages: {ages}")

    if REFERENCE_AGE not in age_files:
        ref_age = ages[0]
        print(f"Warning: Reference age {REFERENCE_AGE} not found. Using {ref_age}.")
    else:
        ref_age = REFERENCE_AGE

    # Load Reference
    reference_img = load_cv2_image(age_files[ref_age])
    
    embeddings = []
    print(f"Preprocessing with Histogram Matching (Ref: Age {ref_age})...")
    for age in ages:
        target_img = load_cv2_image(age_files[age])
        if age == ref_age:
            matched_img = reference_img
        else:
            matched_img = match_histograms(target_img, reference_img, channel_axis=-1)
        
        print(f"Extracting embedding for Age {age}...")
        emb = get_cls_embedding_from_numpy(matched_img, processor, model, device, input_size=INPUT_SIZE)
        embeddings.append(emb)
    
    embeddings = np.array(embeddings)
    print(f"Extracted {len(embeddings)} embeddings.")
    
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    print(f"PCA result shape: {embeddings_2d.shape}")
    
    # Consistency Calculation: First vs Last as main vector
    emb_first = embeddings[0]
    emb_last = embeddings[-1]
    v_total = (emb_last - emb_first).reshape(1, -1)
    
    print("\n--- Aging Consistency Audit (Color Normalized) ---")
    for i in range(1, len(embeddings) - 1):
        emb_mid = embeddings[i]
        v_step = (emb_mid - emb_first).reshape(1, -1)
        sim = cosine_similarity(v_total, v_step)[0][0]
        print(f"Sim (Age {ages[0]}->{ages[-1]} vs {ages[0]}->{ages[i]}): {sim:.4f}")

if __name__ == "__main__":
    main()
