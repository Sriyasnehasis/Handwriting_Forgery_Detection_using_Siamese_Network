import pickle
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load pairs
print("📥 Loading pairs from pickle file...")
with open('data/processed/sig_pairs_labels.pkl', 'rb') as f:
    data = pickle.load(f)

pairs = data['pairs']
labels = data['labels']

print(f"✅ Loaded {len(pairs)} pairs")
print(f"   Positives: {sum(labels)}, Negatives: {len(labels)-sum(labels)}")

# Preprocessing function
def preprocess_signature(image_path):
    """Preprocess signature image for neural network"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"⚠️ Warning: Could not load {image_path}")
        return np.zeros((224, 224, 1), dtype=np.float32)
    
    # Resize to 224x224
    img = cv2.resize(img, (224, 224))
    
    # Apply binary threshold
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Normalize to [0, 1]
    img = img.astype('float32') / 255.0
    
    # Add channel dimension
    img = img[..., np.newaxis]
    
    return img

# Process all pairs
print("\n🔧 Preprocessing all image pairs...")
X1 = []
X2 = []
y = []

for (img1_path, img2_path), label in tqdm(zip(pairs, labels), total=len(pairs)):
    processed_img1 = preprocess_signature(img1_path)
    processed_img2 = preprocess_signature(img2_path)
    
    X1.append(processed_img1)
    X2.append(processed_img2)
    y.append(label)

# Convert to numpy arrays
X1 = np.array(X1)
X2 = np.array(X2)
y = np.array(y)

print(f"\n✅ Preprocessing complete!")
print(f"   X1 shape: {X1.shape}")
print(f"   X2 shape: {X2.shape}")
print(f"   y shape: {y.shape}")

# Split into train, validation, and test sets
print("\n📊 Splitting data into train/val/test...")

# First split: 80% train, 20% temp (for val+test)
X1_train, X1_temp, X2_train, X2_temp, y_train, y_temp = train_test_split(
    X1, X2, y, test_size=0.2, random_state=42, stratify=y
)

# Second split: Split temp into 50% val, 50% test (10% each of total)
X1_val, X1_test, X2_val, X2_test, y_val, y_test = train_test_split(
    X1_temp, X2_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"✅ Data split complete!")
print(f"   Training: {len(y_train)} pairs ({sum(y_train)} positive, {len(y_train)-sum(y_train)} negative)")
print(f"   Validation: {len(y_val)} pairs ({sum(y_val)} positive, {len(y_val)-sum(y_val)} negative)")
print(f"   Test: {len(y_test)} pairs ({sum(y_test)} positive, {len(y_test)-sum(y_test)} negative)")

# Save processed data
print("\n💾 Saving processed data...")
np.savez_compressed('data/processed/train_data.npz',
                   X1=X1_train, X2=X2_train, y=y_train)
np.savez_compressed('data/processed/val_data.npz',
                   X1=X1_val, X2=X2_val, y=y_val)
np.savez_compressed('data/processed/test_data.npz',
                   X1=X1_test, X2=X2_test, y=y_test)

print("✅ All data saved successfully!")
print("\n🎯 Ready for model training!")
