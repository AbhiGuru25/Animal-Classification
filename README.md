# Animal Classification Web App 🐾

An interactive Streamlit web application that uses Deep Learning (CNNs with Transfer Learning) to automatically identify 15 distinct animal species from uploaded images. Built as part of the Unified Mentor Internship Project.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)

## 🎯 Objective
Build a computer vision system that can automatically classify the animal species in a given image. The system classifies images into 15 distinct animal categories (`Bear`, `Bird`, `Cat`, `Cow`, `Deer`, `Dog`, `Dolphin`, `Elephant`, `Giraffe`, `Horse`, `Kangaroo`, `Lion`, `Panda`, `Tiger`, `Zebra`), useful for wildlife monitoring, education, and ecological research.

## ✨ Features
- **Real-Time Image Classification:** Upload any JPG/PNG image and get an instant prediction.
- **Top 5 Confidence Chart:** Interactive Plotly bar chart showing the model's confidence across the top 5 most likely species.
- **Model Performance Metrics:** Embedded dashboard displaying accuracy, precision, and recall metrics (~92% Accuracy on the test set).
- **Responsive UI:** Modern, clean sidebar navigation and glassmorphism-inspired design.

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/abhiguru25/animal-classification.git
cd animal-classification/"Animal Classification"
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

## 🧠 Model Architecture
- **Base Model:** Convolutional Neural Network (CNN) with Transfer Learning.
- **Input Shape:** `224 × 224 × 3` RGB images.
- **Optimization:** Categorical Cross-Entropy Loss, Adam Optimizer.

## 📁 Repository Structure
```
├── app.py                            # Main Streamlit application file
├── requirements.txt                  # Python dependencies
├── animal_classification_model.h5    # Pre-trained Keras model (ensure this is present)
└── README.md                         # This documentation
```

---
*Developed by Abhivirani | Unified Mentor Internship Project*
