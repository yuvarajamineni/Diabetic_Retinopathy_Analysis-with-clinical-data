import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = 'model/diabetic_retinopathy_model.keras'
IMG_SIZE = (150, 150)

model = load_model(MODEL_PATH)
def preprocess_image(image_path):
    image = load_img(image_path, target_size=IMG_SIZE)
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image
def preprocess_clinical_data(age, sex, bmi, glucose, blood_pressure):
    clinical_data = np.array([[age, sex, bmi, glucose, blood_pressure]])
    return clinical_data
def evaluate(image_path, age, sex, bmi, glucose, blood_pressure):
    image = preprocess_image(image_path)
    clinical_data = preprocess_clinical_data(age, sex, bmi, glucose, blood_pressure)
    prediction = model.predict([image, clinical_data])
    predicted_label = np.argmax(prediction, axis=1)[0]
    return predicted_label
image_path = 'data/raw/train_images/1b32e1d775ea.png'
age = 55
sex = 1
bmi = 24.5
glucose = 120
blood_pressure = 140
predicted_label = evaluate(image_path, age, sex, bmi, glucose, blood_pressure)
print(f'Predicted label: {predicted_label}')
