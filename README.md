# Skin Detection Project

This project provides a machine learning–based solution for **skin disease detection** using image classification.  
It includes code for training the model, running a Flask web app, and generating predictions through a simple UI.

---

## 🚀 Features

- 🧠 Machine Learning model for skin condition classification  
- 📷 Image-based prediction using a trained deep learning model  
- 🌐 Flask Web Application for user-friendly interaction  
- 🛠️ Training script to build or retrain the model  
- 📄 Report summarizing methodology and results  
- 📦 Requirements file for installing dependencies  

---

## 📂 Project Structure

skin-detect/
├── app.py # Flask web app for prediction
├── train_model.py # Script to train the ML model
├── requirements.txt # Python dependencies
├── report.pdf # Project documentation/report
├── run_app.bat # Start the web app (Windows)
└── .gitignore # Ignored files/folders

yaml
Copy code

---

## 🛠️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Sindhu9ly/ISE_40.git
cd ISE_40
2️⃣ Create & Activate Virtual Environment
bash
Copy code
python -m venv venv
venv\Scripts\activate    # Windows
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
▶️ Running the Application
Option 1 — Using Python
bash
Copy code
python app.py
Option 2 — Using the Batch File
Double–click:

Copy code
run_app.bat
Then open your browser and visit:

cpp
Copy code
http://127.0.0.1:5000/
Upload an image → Get predictions.

🧠 Training the Model
To retrain the model:

bash
Copy code
python train_model.py
Ensure the dataset is arranged properly before training.
