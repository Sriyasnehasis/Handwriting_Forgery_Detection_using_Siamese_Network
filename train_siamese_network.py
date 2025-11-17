import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ========================== 1. LOAD THE DATA ==========================
print("📥 Loading data...")
train = np.load('data/processed/train_data.npz')
val = np.load('data/processed/val_data.npz')
test = np.load('data/processed/test_data.npz')

X1_train, X2_train, y_train = train['X1'], train['X2'], train['y']
X1_val, X2_val, y_val = val['X1'], val['X2'], val['y']
X1_test, X2_test, y_test = test['X1'], test['X2'], test['y']

print("✅ Data loaded.")
print(f"Train: {X1_train.shape}, {X2_train.shape}, {y_train.shape}")
print(f"Val:   {X1_val.shape}, {X2_val.shape}, {y_val.shape}")
print(f"Test:  {X1_test.shape}, {X2_test.shape}, {y_test.shape}")

# =================== 2. DEFINE SIAMESE NETWORK COMPONENTS =============

def build_base_cnn(input_shape=(224,224,1)):
    inp = Input(shape=input_shape)
    
    x = Conv2D(64, (3,3), activation='relu', padding='same')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.2)(x)
    
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.2)(x)
    
    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.3)(x)
    
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    return Model(inp, x, name='Base_CNN')

def euclidean_distance(vectors):
    x, y = vectors
    sum_square = K.sum(K.square(x-y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

# ===================== 3. BUILD FULL SIAMESE MODEL ====================

input_shape = (224, 224, 1)
base_cnn = build_base_cnn(input_shape)

input_a = Input(shape=input_shape, name='input_a')
input_b = Input(shape=input_shape, name='input_b')

processed_a = base_cnn(input_a)
processed_b = base_cnn(input_b)

distance = Lambda(euclidean_distance)([processed_a, processed_b])
output = Dense(1, activation='sigmoid')(distance)  # Probability of being "same" (genuine)

siamese_model = Model(inputs=[input_a, input_b], outputs=output)
siamese_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

siamese_model.summary()

# ======================== 4. TRAIN THE MODEL ==========================

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('models/best_siamese_model.h5', save_best_only=True, monitor='val_loss'),
    ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
]

print("\n🚀 Training the Siamese network...")
history = siamese_model.fit([X1_train, X2_train], y_train,
                            validation_data=([X1_val, X2_val], y_val),
                            batch_size=32, epochs=30, callbacks=callbacks)

# ======================== 5. EVALUATE THE MODEL =======================

print("\n📊 Evaluating on test set...")
test_loss, test_acc = siamese_model.evaluate([X1_test, X2_test], y_test)
print(f"\n✅ Test loss: {test_loss:.4f}   Test accuracy: {test_acc:.4f}")

# Optional: Predict and analyze results
from sklearn.metrics import classification_report, confusion_matrix

y_pred_prob = siamese_model.predict([X1_test, X2_test])
y_pred = (y_pred_prob > 0.5).astype(int)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ================== 6. SAVE THE FINAL MODEL ===================

siamese_model.save('models/final_siamese_model.h5')
print("\n✅ Final model saved to models/final_siamese_model.h5")

