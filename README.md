# **Potato Disease Classification using Deep Learning**

## **Overview**

This project implements a deep learning model to classify **Potato Diseases**, specifically **Early Blight** and **Late Blight**, using **Convolutional Neural Networks (CNNs)**. The model is trained on the **PlantVillage Dataset** and achieves high accuracy in disease classification.

## **Key Features**

- ✅ **CNN-based Deep Learning Model**
- ✅ **Classification of Early Blight and Late Blight in Potatoes**
- ✅ **Image Augmentation for Better Generalization**
- ✅ **Trained on PlantVillage Dataset**
- ✅ **Achieves High Accuracy with TensorFlow & Keras**

## **Dataset**

The dataset is sourced from **PlantVillage** and consists of:

- 📌 **Early Blight Images**
- 📌 **Late Blight Images**
- 📌 **Healthy Potato Leaf Images**

## **Model Architecture**

The model follows a **CNN-based approach**:

- **Conv2D Layers** with ReLU activation for feature extraction
- **MaxPooling Layers** to reduce spatial dimensions
- **Fully Connected Dense Layers** for classification
- **Softmax Activation** for multi-class classification

## **Implementation Details**

- **Programming Language**: Python (TensorFlow, Keras)
- **Libraries Used**: NumPy, OpenCV, Matplotlib, ImageDataGenerator
- **Model Input Size**: 224x224 pixels
- **Loss Function**: Categorical Cross-Entropy
- **Optimizer**: Adam
- **Batch Size**: 32
- **Epochs**: 20

## **Installation & Usage**

### **1️⃣ Clone the Repository**

```bash
git clone https://github.com/gupta-adityaa/PotatoDiseaseClassification.git
cd PotatoDiseaseClassification
```

### **2️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3️⃣ Predict Disease from an Image**

```python
from tensorflow.keras.preprocessing import image
import numpy as np

def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array
```

## **Future Scope**

- 🚀 **Improve accuracy with more advanced architectures (e.g., EfficientNet, ResNet)**
- 🚀 **Deploy as a Web-based App for farmers**
- 🚀 **Expand the dataset for better generalization**



