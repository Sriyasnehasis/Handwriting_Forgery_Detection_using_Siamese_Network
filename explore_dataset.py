import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Dataset paths
dataset_path = "data/raw/cedardataset"
genuine_path = os.path.join(dataset_path, "genuine")
forged_path = os.path.join(dataset_path, "forged")

# Explore dataset
print("🔍 EXPLORING CEDAR DATASET")
print("=" * 40)

# Count files
if os.path.exists(genuine_path):
    genuine_count = len(os.listdir(genuine_path))
    print(f"✅ Genuine signatures: {genuine_count}")
else:
    print("❌ Genuine folder not found")

if os.path.exists(forged_path):
    forged_count = len(os.listdir(forged_path))
    print(f"✅ Forged signatures: {forged_count}")
else:
    print("❌ Forged folder not found")

print(f"📊 Total signatures: {genuine_count + forged_count}")




def load_and_display_signature(image_path, title="Signature"):
    """Load and display a signature image"""
    
    # Load image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return None
    
    # Convert BGR to RGB (OpenCV uses BGR, matplotlib uses RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display image info
    print(f"📷 Image: {title}")
    print(f"   Size: {image_rgb.shape}")
    print(f"   Type: {image_rgb.dtype}")
    
    # Display image
    plt.figure(figsize=(8, 4))
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()
    
    return image_rgb

# Test with sample images
print("\n🖼️ LOADING SAMPLE SIGNATURES")
print("=" * 40)

# Load a genuine signature
genuine_files = os.listdir(genuine_path)
if genuine_files:
    sample_genuine = os.path.join(genuine_path, genuine_files[0])
    load_and_display_signature(sample_genuine, "Sample Genuine Signature")

# Load a forged signature
forged_files = os.listdir(forged_path)
if forged_files:
    sample_forged = os.path.join(forged_path, forged_files[0])
    load_and_display_signature(sample_forged, "Sample Forged Signature")




def preprocess_signature_basic(image):
    """Basic preprocessing for signature images"""
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Resize to standard size
    resized = cv2.resize(gray, (224, 224))
    
    # Apply binary threshold
    _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY_INV)
    
    return resized, binary

def compare_preprocessing_steps(image_path):
    """Show preprocessing steps side by side"""
    
    # Load original
    original = cv2.imread(image_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    resized, binary = preprocess_signature_basic(original_rgb)
    
    # Display comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(resized, cmap='gray')
    axes[1].set_title('Resized (224x224)')
    axes[1].axis('off')
    
    axes[2].imshow(binary, cmap='gray')
    axes[2].set_title('Binary Threshold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return original_rgb, resized, binary

# Test preprocessing
print("\n🔧 TESTING PREPROCESSING")
print("=" * 40)

if genuine_files:
    sample_path = os.path.join(genuine_path, genuine_files[0])
    orig, resized, binary = compare_preprocessing_steps(sample_path)
    
    print(f"✅ Original shape: {orig.shape}")
    print(f"✅ Resized shape: {resized.shape}")
    print(f"✅ Binary shape: {binary.shape}")
