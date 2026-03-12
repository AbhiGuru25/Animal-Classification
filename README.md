# Animal Image Classification 🐾

## Objective
The objective of this project is to build a system capable of identifying an animal from an image into one of 15 different categories.

## Dataset
The dataset contains images of the following 15 animal classes:
Bear, Bird, Cat, Cow, Deer, Dog, Dolphin, Elephant, Giraffe, Horse, Kangaroo, Lion, Panda, Tiger, Zebra.

## Features
*   **Machine Learning Model:** Utilizes Transfer Learning with the MobileNetV2 architecture (pre-trained on ImageNet) to extract features and classify animal images effectively.
*   **Web Interface:** An interactive Streamlit web application where users can upload a picture of an animal to receive a species prediction.

## How to Run Locally

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## Files in this Repository
*   `Animal_Classification.ipynb`: The Jupyter Notebook containing data augmentation, model configuration, and training.
*   `app.py`: The Streamlit web application.
*   `animal_classification_model.h5`: The saved trained MobileNetV2 classification model.
*   `Animal_Classification_Report.pdf`: The detailed project report.
*   `requirements.txt`: Python dependencies.
