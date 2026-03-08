# Handwriting Forgery Detection using Siamese Networks

A deep learning project designed to distinguish between genuine and forged signatures using **Siamese Neural Networks**. This model learns to identify unique handwriting characteristics and verify authenticity by comparing pairs of signatures.

##  Project Overview
Handwriting verification is a critical task in forensics and banking. This project implements a Siamese network architecture that processes two input images and computes a similarity score. By using **Contrastive Loss**, the model is trained to minimize the distance between genuine signatures and maximize the distance between a genuine signature and a forgery.

##  Key Features
* **Siamese Architecture:** Twin Convolutional Neural Networks (CNNs) share weights to extract robust feature vectors from signature pairs.
* **Automated Data Pipeline:** A custom `SiameseDataGenerator` handles real-time data loading and augmentation to prevent overfitting.
* **Smart Training:** Features automated callbacks like `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint` to ensure the best model is saved.
* **Cloud Ready:** Fully optimized for Google Colab with seamless Drive integration for handling large datasets.

##  Technical Stack
* **Language:** Python
* **Deep Learning:** TensorFlow / Keras
* **Data Processing:** NumPy, OpenCV
* **Visualization:** Matplotlib

##  Project Structure
* `Handwriting_Forgery_Detection.ipynb`: The main notebook containing data preprocessing, model architecture, and training loops.
* **Data Handling:** Uses `.npz` formatted datasets for efficient storage and loading.
* **Models:** Automatically saves the best-performing model as `best_siamese_contrastive.h5`.

##  Setup & Usage
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Sriyasnehasis/Handwriting_Forgery_Detection_using_Siamese_Network.git](https://github.com/Sriyasnehasis/Handwriting_Forgery_Detection_using_Siamese_Network.git)
    ```
2.  **Dataset Placement:**
    Upload your signature dataset (e.g., `train_data.npz` and `test_data.npz`) to your Google Drive at the following path:
    `/content/drive/MyDrive/Signature Dataset/`
3.  **Run in Colab:**
    Open the notebook and follow the cells to mount your drive and begin training.

##  Training Configuration
* **Batch Size:** 16
* **Loss Function:** Contrastive Loss
* **Optimizer:** Adaptive learning rate with reduction on plateau
* **Early Stopping:** Monitoring validation loss with a patience of 12 epochs
