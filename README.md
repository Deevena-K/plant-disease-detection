# 🌱 Plant Disease Detection Using CNN

A deep learning-based **Plant Disease Detection System** that identifies plant diseases from leaf images using a **Convolutional Neural Network (CNN)** built with TensorFlow and Keras.

The model is trained on the **New Plant Diseases Dataset (Augmented)** and can classify leaf images into **38 different plant disease and healthy categories**.

---

## 📌 Project Overview

Plant diseases can significantly affect crop production and quality. Early identification of diseases can help farmers take appropriate action and reduce crop losses.

This project uses **Convolutional Neural Networks (CNNs)** to automatically analyze plant leaf images and predict whether the plant is healthy or affected by a particular disease.

The complete workflow includes:

* Dataset extraction from Google Drive
* Image preprocessing
* Data augmentation
* Training and validation data generation
* CNN model construction
* Model training
* Performance evaluation
* Accuracy, precision, recall and loss analysis
* Saving the trained model for future predictions

---

## 🎯 Objectives

* Detect plant diseases from leaf images.
* Classify images into 38 different categories.
* Build a CNN model using TensorFlow/Keras.
* Evaluate the model using accuracy, precision, recall and loss.
* Save the trained model for future deployment and inference.

---

## 📊 Dataset

The project uses the **New Plant Diseases Dataset (Augmented)**.

### Dataset Statistics

| Dataset    |    Images |
| ---------- | --------: |
| Training   |    63,282 |
| Validation |     1,742 |
| Classes    |        38 |
| Image Size | 224 × 224 |
| Color Mode |       RGB |

The images are resized to **224 × 224 pixels** and normalized using:

```python
rescale=1/255.0
```

---

## 🌿 Supported Classes

The model recognizes the following 38 categories:

1. Apple - Apple Scab
2. Apple - Black Rot
3. Apple - Cedar Apple Rust
4. Apple - Healthy
5. Blueberry - Healthy
6. Cherry - Powdery Mildew
7. Cherry - Healthy
8. Corn (Maize) - Cercospora Leaf Spot / Gray Leaf Spot
9. Corn (Maize) - Common Rust
10. Corn (Maize) - Northern Leaf Blight
11. Corn (Maize) - Healthy
12. Grape - Black Rot
13. Grape - Esca (Black Measles)
14. Grape - Leaf Blight
15. Grape - Healthy
16. Orange - Huanglongbing (Citrus Greening)
17. Peach - Bacterial Spot
18. Peach - Healthy
19. Pepper Bell - Bacterial Spot
20. Pepper Bell - Healthy
21. Potato - Early Blight
22. Potato - Late Blight
23. Potato - Healthy
24. Raspberry - Healthy
25. Soybean - Healthy
26. Squash - Powdery Mildew
27. Strawberry - Leaf Scorch
28. Strawberry - Healthy
29. Tomato - Bacterial Spot
30. Tomato - Early Blight
31. Tomato - Late Blight
32. Tomato - Leaf Mold
33. Tomato - Septoria Leaf Spot
34. Tomato - Spider Mites
35. Tomato - Target Spot
36. Tomato - Tomato Yellow Leaf Curl Virus
37. Tomato - Tomato Mosaic Virus
38. Tomato - Healthy

---

## 🛠️ Technologies Used

* **Python**
* **TensorFlow**
* **Keras**
* **NumPy**
* **Pandas**
* **OpenCV**
* **Matplotlib**
* **Seaborn**
* **Google Colab**
* **Google Drive**

---

## 🧠 Model Architecture

A custom CNN architecture is used for classification.

### Architecture

```text
Input Image
224 × 224 × 3
      ↓
Conv2D - 32 Filters - 7×7
      ↓
MaxPooling2D
      ↓
Conv2D - 64 Filters - 5×5
      ↓
MaxPooling2D
      ↓
Conv2D - 128 Filters - 3×3
      ↓
Conv2D - 256 Filters - 3×3
      ↓
MaxPooling2D
      ↓
Flatten
      ↓
Dense - 128 neurons
      ↓
Dense - 64 neurons
      ↓
Dense - 38 neurons
      ↓
Softmax
      ↓
Predicted Plant Disease
```

### Model Parameters

* **Total Parameters:** 26,125,990
* **Trainable Parameters:** 26,125,990
* **Input Shape:** `(224, 224, 3)`
* **Output Classes:** `38`
* **Output Activation:** `Softmax`
* **Loss Function:** `Categorical Crossentropy`
* **Optimizer:** Adam

---

## 🔄 Image Preprocessing

Images are resized to:

```text
224 × 224
```

Pixel values are normalized from `[0, 255]` to `[0, 1]`:

```python
rescale=1/255.0
```

For training, rotation augmentation is also applied:

```python
rotation_range=90
```

The training data uses categorical labels because the model performs **multi-class classification**.

---

## ⚙️ Training Configuration

The model was trained using:

```python
optimizer = Adam
loss = categorical_crossentropy
epochs = 8
batch_size = 164
```

The following callbacks were used:

### Early Stopping

Stops training when validation loss stops improving.

```python
EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)
```

### Model Checkpoint

Saves the model with the best validation loss.

```python
ModelCheckpoint(
    'best_model.keras',
    monitor='val_loss',
    save_best_only=True
)
```

### Reduce Learning Rate

Reduces the learning rate when validation loss stops improving.

```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=5,
    min_lr=0.000001
)
```

---

## 📈 Training Results

The model showed significant improvement during training.

| Epoch | Training Accuracy | Validation Accuracy |
| ----: | ----------------: | ------------------: |
|     1 |            20.98% |              60.85% |
|     2 |            67.39% |              69.69% |
|     3 |            81.14% |              84.16% |
|     4 |            86.34% |              85.42% |
|     5 |            89.69% |              87.77% |
|     6 |            91.16% |              89.61% |
|     7 |        **93.42%** |          **91.16%** |
|     8 |            94.26% |              89.95% |

The best validation accuracy was achieved in **Epoch 7**, with approximately **91.16% validation accuracy**.

---

## 🧪 Model Evaluation

The trained model was evaluated using the test generator.

### Final Evaluation Results

| Metric    |      Score |
| --------- | ---------: |
| Loss      | **0.2416** |
| Accuracy  | **92.17%** |
| Precision | **93.53%** |
| Recall    | **91.16%** |

### Summary

The CNN achieved:

* 🎯 **92.17% Accuracy**
* 🎯 **93.53% Precision**
* 🎯 **91.16% Recall**

These results demonstrate that the model can effectively classify plant leaf images across the 38 supported categories.

---

## 💾 Saving the Model

The trained model is saved in Keras format:

```python
model.save('PDDS.keras')
```

The model can later be loaded using:

```python
from tensorflow import keras

model = keras.models.load_model('PDDS.keras')
```

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

### 2. Install Dependencies

```bash
pip install tensorflow numpy pandas matplotlib seaborn opencv-python
```

### 3. Open the Notebook

The project can be run using:

* Google Colab
* Jupyter Notebook
* JupyterLab

### 4. Dataset Setup

Upload the dataset to Google Drive and update the dataset path in the notebook.

Example:

```python
train = '/content/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'
```

### 5. Train the Model

Run the notebook cells sequentially to:

1. Mount Google Drive
2. Extract the dataset
3. Load and preprocess images
4. Create training and validation generators
5. Build the CNN
6. Train the model
7. Evaluate the model
8. Save the trained model

---

## 🔮 Making Predictions

After loading the trained model, an input image can be resized to `224 × 224`, normalized, and passed to the model.

Example:

```python
import cv2
import numpy as np
from tensorflow import keras

model = keras.models.load_model("PDDS.keras")

image = cv2.imread("leaf.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))

image = image / 255.0
image = np.expand_dims(image, axis=0)

prediction = model.predict(image)

predicted_class = np.argmax(prediction[0])
confidence = np.max(prediction[0])

print("Predicted Class:", predicted_class)
print("Confidence:", confidence)
```

The predicted class index can be mapped to the corresponding disease name using the class mapping created by the dataset generator.

---

## 📁 Suggested Project Structure

```text
Plant-Disease-Detection/
│
├── Plant_Disease_Detection.ipynb
├── PDDS.keras
├── best_model.keras
├── README.md
│
└── dataset/
    ├── train/
    └── valid/
```

> The complete dataset is not included in the repository because of its size.

---

## 📊 Evaluation Metrics

The project uses multiple metrics to evaluate model performance.

### Accuracy

Measures the percentage of correctly classified images.

### Precision

Measures how many of the images predicted as a particular class were actually from that class.

### Recall

Measures how many actual samples of a class were correctly detected.

### Loss

Categorical cross-entropy is used as the classification loss function.

---

## 🔍 Key Features

* 🌱 38-class plant disease classification
* 🖼️ Image-based disease detection
* 🧠 Custom CNN architecture
* 🔄 Image augmentation
* 📊 Accuracy, precision and recall evaluation
* 💾 Keras model saving
* 📈 Training and validation performance visualization
* ☁️ Google Colab and Google Drive support

---

## ⚠️ Limitations

* The model is trained on a fixed set of 38 classes.
* Performance may vary with real-world images that differ significantly from the training dataset.
* Images with poor lighting, complex backgrounds, or severe occlusion may reduce prediction accuracy.
* The model should be considered a **decision-support tool**, not a replacement for professional agricultural diagnosis.

---

## 🔮 Future Improvements

Possible improvements include:

* Use transfer learning models such as **EfficientNet, MobileNet, ResNet, or InceptionV3**.
* Add more real-world plant images.
* Improve data augmentation.
* Use **Grad-CAM** for visual explanations of predictions.
* Build a web interface using **Flask, FastAPI, or Streamlit**.
* Deploy the model as a cloud-based application.
* Add confidence-based prediction warnings.
* Support additional crops and diseases.
* Optimize the model for mobile and edge-device deployment.

---

## 👩‍💻 Author

**Lahitha Valluri**

B.Tech – Information Technology

---

## 📄 License

This project is intended for educational and research purposes.

---

## ⭐ Acknowledgements

* TensorFlow / Keras for the deep learning framework
* The New Plant Diseases Dataset (Augmented) for the training images
* Google Colab for providing the development and training environment

