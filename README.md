# Facial Emotion Recognition (FER) with CNN-LSTM

Trained on FER-2013 Dataset | Accuracy: \~84%

## 📌 Overview

Facial Emotion Recognition (FER) uses computer vision to identify emotions from human facial expressions. This project implements a hybrid **CNN-LSTM** model trained on the **FER-2013** dataset to classify 7 basic emotions.

### 🔍 Emotions Recognized

* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

---

## 🧠 Model Architecture

```
Input (48x48x1)
↓
CNN Layers (spatial features)
↓
Flatten & Reshape
↓
Bidirectional LSTM (temporal patterns)
↓
Dense Layers
↓
Softmax (7 classes)
```

### Why CNN-LSTM?

* **CNN** extracts spatial features from facial structures.
* **LSTM** captures sequential dependencies from flattened image features.
* **Together**, they improve emotion classification performance.

---

## 📦 Dataset

| Feature | Description                               |
| ------- | ----------------------------------------- |
| Source  | FER-2013 (Kaggle)                         |
| Format  | 48x48 grayscale images (CSV format)       |
| Size    | 35,887 samples                            |
| Splits  | Train (28,709), Val (3,589), Test (3,589) |
| Labels  | Emotion ID (0–6), pixel values, usage     |

---

## ⚙️ Training Details

| Parameter     | Value                                |
| ------------- | ------------------------------------ |
| Optimizer     | Adam                                 |
| Loss Function | Categorical Crossentropy             |
| Epochs        | 50                                   |
| Batch Size    | 64                                   |
| Augmentation  | Rotation, Zoom, Flip                 |
| Oversampling  | RandomOverSampler (imbalanced-learn) |

---

## 📊 Performance

* ✅ Test Accuracy: \~84%
* 📉 Visuals: Confusion matrix, classification report
* 📈 Evaluated using precision, recall, F1-score
---

## 📁 Project Structure

```
📦 FER-CNN-LSTM
 ┣ 📜 fer2013.csv
 ┣ 📓 FER_CNN_LSTM.ipynb
 ┗ 📄 README.md
```

## 👩‍💻 Owners

```
 ┣ 👩‍💻 Lumakang
 ┣ 👩‍💻 Tuazon
 ┗ 👩‍💻 Virtudez
```

