import streamlit as st
import cv2
import numpy as np
import joblib
import sqlite3
from datetime import datetime
import os

# Load the best model
best_model = joblib.load('best_model.pkl')

# Database setup
DB_FILE = "patient_data.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS PatientData (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            phone TEXT,
            address TEXT,
            image BLOB,
            result TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

# Save patient data to the database
def save_to_db(name, age, phone, address, image_blob, result, timestamp):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO PatientData (name, age, phone, address, image, result, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (name, age, phone, address, image_blob, result, timestamp))
    conn.commit()
    conn.close()

# Function to process image and predict
def process_and_predict(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (200, 200)).reshape(1, -1) / 255.0
    prediction = best_model.predict(img_resized)
    return img, "Positive Tumor" if prediction[0] == 1 else "No Tumor"

# Function to search for patient data by phone number
def search_patient_by_phone(phone):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT name, age, phone, address, image, result, timestamp FROM PatientData WHERE phone = ?", (phone,))
    patient = cursor.fetchone()
    conn.close()
    return patient

# Initialize database
init_db()

# Streamlit interface for tab selection
st.title("Brain Tumor Detection System")

# First, select the page to view
page_selection = st.selectbox("Select a Tab", ["Prediction", "Patient Data"])

# Proceed based on the selected page
if page_selection == "Prediction":
    st.title("Brain Tumor Prediction")
    st.write("Fill in the patient details, upload an MRI image, and get a prediction. Data will be saved securely.")

    # Patient details input
    name = st.text_input("Patient Name")
    age = st.number_input("Age", min_value=0, step=1)
    phone = st.text_input("Phone Number")
    address = st.text_area("Address")

    # Image upload
    uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

    if st.button("Predict and Save"):
        if name and age and phone and address and uploaded_file:
            # Display uploaded image
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            st.write("Processing the image...")

            # Process the image and get prediction
            img, result = process_and_predict(uploaded_file)

            # Save image as binary
            _, img_encoded = cv2.imencode('.jpg', img)
            image_blob = img_encoded.tobytes()

            # Record timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Save data to the database
            save_to_db(name, age, phone, address, image_blob, result, timestamp)

            st.write(f"**Prediction:** {result}")
            st.success("Data saved successfully!")
        else:
            st.error("Please fill in all patient details and upload an image.")

elif page_selection == "Patient Data":
    st.title("Search Patient Data")

    # Search by phone number
    search_phone = st.text_input("Enter Patient Phone Number to Search")

    if search_phone:
        patient = search_patient_by_phone(search_phone)

        if patient:
            name, age, phone, address, image_blob, result, timestamp = patient
            # Display patient data
            st.write(f"**Name**: {name}")
            st.write(f"**Age**: {age}")
            st.write(f"**Phone**: {phone}")
            st.write(f"**Address**: {address}")
            st.write(f"**Result**: {result}")
            st.write(f"**Timestamp**: {timestamp}")

            # Convert the image back to display it
            image = cv2.imdecode(np.frombuffer(image_blob, np.uint8), cv2.IMREAD_COLOR)
            st.image(image, caption="Patient MRI Image", use_container_width=True)

        else:
            st.warning("No patient found with that phone number.")
