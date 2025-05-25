from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
IMG_SIZE = (150, 150)
MODEL_PATH = 'model/diabetic_retinopathy_model.keras'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model = load_model(MODEL_PATH)

DR_LABELS = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

def preprocess_image(image_path):
    image = load_img(image_path, target_size=IMG_SIZE)
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def get_treatment_recommendation(dr_stage, clinical_score):
    if dr_stage == 0:
        if clinical_score == 0:
            return "No immediate treatment required. Continue monitoring and maintaining a healthy lifestyle."
        else:
            return "Regular check-ups and lifestyle changes recommended to manage health risks."
    elif dr_stage == 1:
        if clinical_score == 0:
            return "Early stage. Regular monitoring and lifestyle changes recommended."
        else:
            return "Early stage with health risks. Consult a doctor for a personalized treatment plan."
    elif dr_stage == 2:
        if clinical_score == 0:
            return "Moderate stage. Consult a doctor for potential treatments."
        else:
            return "Moderate stage with health risks. Immediate medical consultation recommended."
    elif dr_stage == 3:
        if clinical_score == 0:
            return "Severe stage. Urgent medical treatment required."
        else:
            return "Severe stage with health risks. Intensive treatment and monitoring required."
    elif dr_stage == 4:
        if clinical_score == 0:
            return "Proliferative stage. Immediate medical treatment required."
        else:
            return "Proliferative stage with severe health risks. Advanced medical intervention required."
    else:
        return "Unknown stage. Consult a healthcare provider for more information."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('error', message='No file part'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('error', message='No selected file'))
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        age = request.form.get('age', type=int)
        sex = request.form.get('sex', type=int)
        bmi = request.form.get('bmi', type=float)
        glucose = request.form.get('glucose', type=float)
        blood_pressure = request.form.get('blood_pressure', type=float)

        image = preprocess_image(file_path)
        prediction = model.predict(image)
        predicted_label = np.argmax(prediction, axis=1)[0]

        print(f"Prediction array: {prediction}")
        print(f"Predicted label: {predicted_label}")

        clinical_score = 0
        if glucose > 140 or blood_pressure > 130:
            clinical_score += 1
        if bmi > 30:
            clinical_score += 1

        risk_level = "Low" if clinical_score == 0 else "High"

        suggestions = []
        if clinical_score == 0:
            suggestions.append("Regular check-ups with your doctor are recommended.")
            suggestions.append("Follow a healthy diet to keep your diabetes under control.")
            suggestions.append("Stay active to maintain overall health.")
        elif clinical_score == 1:
            suggestions.append("Monitor your blood glucose and pressure regularly.")
            suggestions.append("Consider consulting a nutritionist for dietary adjustments.")
            suggestions.append("Incorporate moderate exercise into your routine.")
        else:
            suggestions.append("Seek immediate medical attention to manage your blood glucose and pressure.")
            suggestions.append("Consult with a specialist to develop a comprehensive treatment plan.")
            suggestions.append("Strictly adhere to prescribed medications and lifestyle modifications.")
        if predicted_label == 0:
            diagnosis_message = "Your eyes are not infected with Diabetic Retinopathy."
        elif predicted_label == 2:
            diagnosis_message = "Your eyes are infected with Diabetic Retinopathy."
        else:
            diagnosis_message = f"Predicted Diabetic Retinopathy Stage: {DR_LABELS[predicted_label]}"

        treatment_recommendation = get_treatment_recommendation(predicted_label, clinical_score)

        return render_template('result.html', diagnosis_message=diagnosis_message, clinical_score=clinical_score, risk_level=risk_level, suggestions=suggestions, treatment_recommendation=treatment_recommendation)

@app.route('/result')
def result():
    diagnosis_message = request.args.get('diagnosis_message')
    clinical_score = request.args.get('clinical_score')
    risk_level = request.args.get('risk_level')
    suggestions = request.args.get('suggestions')
    treatment_recommendation = request.args.get('treatment_recommendation')
    return render_template('result.html', diagnosis_message=diagnosis_message, clinical_score=clinical_score, risk_level=risk_level, suggestions=suggestions, treatment_recommendation=treatment_recommendation)

@app.route('/error')
def error():
    message = request.args.get('message')
    return render_template('error.html', message=message)

if __name__ == '__main__':
    app.run(debug=True)
