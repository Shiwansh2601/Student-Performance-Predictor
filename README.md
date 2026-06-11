# 🎓 Student Performance Predictor

An intelligent **Machine Learning web application** that predicts a student’s **Performance Index** based on academic and lifestyle-related factors. This project uses a trained ML model and an interactive **Streamlit interface** for real-time predictions.

## 🚀 Live Demo

🔗 **Try the Deployed App:**
https://shiwansh2601-student-performance-predictor-app-d8tlfp.streamlit.app/

---

## 📌 Project Overview

The **Student Performance Predictor** estimates a student's performance score by analyzing important factors that influence academic outcomes.

### Input Features:

* 📚 **Hours Studied**
* 📝 **Previous Scores**
* 😴 **Sleep Hours**
* 🎯 **Extracurricular Activities**
* 📄 **Sample Question Papers Practiced**

The model predicts a **Performance Index** based on these inputs.

---

## ✨ Features

✅ Real-time student performance prediction
✅ Interactive and user-friendly Streamlit UI
✅ Machine Learning-powered predictions
✅ Data preprocessing and model training
✅ Clean deployment using Streamlit Cloud

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Libraries & Frameworks

* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Matplotlib
* Seaborn

### Machine Learning Model

* Random Forest Regressor

---

## 📂 Project Structure

```text
student-performance-predictor/
│── app.py
│── train_model.py
│── Student_Performance.csv
│── student_model.pkl
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone <your-repository-link>
cd student-performance-predictor
```

### 2️⃣ Create a Virtual Environment (Recommended)

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Train the Model

```bash
python train_model.py
```

### 5️⃣ Run the Streamlit Application

```bash
streamlit run app.py
```

---

## 📊 Model Workflow

1. Load and preprocess student dataset
2. Train the Machine Learning model
3. Save the trained model (`student_model.pkl`)
4. Use Streamlit UI for prediction inputs
5. Generate performance predictions instantly

---

## 🌐 Deployment

This project is deployed using **Streamlit Cloud**.

🔗 **Live Application:**
https://shiwansh2601-student-performance-predictor-app-d8tlfp.streamlit.app/

---

## 👨‍💻 Author

**Shiwansh Tiwari**
Aspiring AI/ML Engineer | Data Science Enthusiast

---

⭐ If you found this project useful, consider giving it a **star** on GitHub!
