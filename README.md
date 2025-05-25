# Diabetic Retinopathy Detection with Clinical Data Integration 🩺📊
This project focuses on detecting Diabetic Retinopathy (DR) using a pre-trained deep learning model and clinical data integration for personalized predictions and treatment recommendations. The app is built using Flask and provides a user-friendly interface to upload images and enter clinical data.

## Features 🌟
DR Stage Prediction: Detects the severity of DR (No DR, Mild, Moderate, Severe, Proliferative) from fundus images.
Clinical Data Integration: Improves prediction accuracy by combining image data with clinical inputs (age, BMI, blood glucose, etc.).
Personalized Treatment Suggestions: Based on DR predictions and clinical features.
Web Interface: A Flask-based web application for image upload and clinical data input.

## Dataset 📊
You can access the dataset used for training the model from Kaggle:

[Aptos 2019 Blindness Detection Dataset](https://www.kaggle.com/c/aptos2019-blindness-detection)

The dataset includes fundus images labeled with different stages of DR, which are used to train the deep learning model.

## Project Structure 📂

![image](https://github.com/user-attachments/assets/c5b0f4d9-0f56-4ca8-b018-5d185a234b72)

## How to Run 🚀
Follow the steps below to run the project locally.

**Step 1: Clone the Repository**

sh

     git clone https://github.com/DeepakGowda-Official/diabetic-retinopathy-clinical.git
     cd diabetic-retinopathy-clinical

**Step 2: Install the Dependencies**

Set up a Python virtual environment and install all required packages:

sh

     python3 -m venv venv
     source venv/bin/activate    # On Windows use `venv\Scripts\activate`
     pip install -r requirements.txt

**Step 3: Download the Dataset**

Download the dataset from Kaggle using the following link and place it in the data/raw/ folder:

[Aptos 2019 Blindness Detection Dataset](https://www.kaggle.com/c/aptos2019-blindness-detection)

The training images should be placed inside the train_images/ directory, and the train.csv file should be placed in the raw/ folder.

**Step 4: Run the Flask Web Application**

Launch the Flask application by running the following command:

sh

     python app/app.py

Visit http://127.0.0.1:5000/ in your browser to access the web interface.

**Step 5: Using the Application**

 1. Upload a Fundus Image: Upload an image of the eye for DR prediction.
 2. Input Clinical Data: Enter clinical data such as age, BMI, glucose level, etc.
 3. Submit: Click submit to receive the predicted DR stage and personalized treatment     suggestions.

## Retrain the Model 🧑‍💻
If you'd like to retrain the model using the training script:
sh

     python train_model.py
     
This will use the dataset in the data/raw/ folder to retrain the model, which will then be saved to the model/ folder as diabetic_retinopathy_model.keras.

## Predict & Evaluate the Model 🎯

To evaluate the model or make predictions using a new set of images, use the following command:

sh

     python evaluate.py
This script will output the model's performance metrics and any predictions based on the input data.


  # Transforming sight and saving lives—your code makes a real difference! 👁️✨
  
