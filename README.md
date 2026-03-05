# 🎓 Student Career Prediction System

A Machine Learning project that predicts the **most suitable career path for a student** based on their skills, academic performance, and background.

This project uses **Python, Scikit-Learn, and Streamlit** to build a predictive model and deploy it as an interactive web application.

---

# 📌 Project Overview

Choosing the right career is difficult for many students.
This project uses **Machine Learning algorithms** to analyze student data and recommend a potential career path.

The system evaluates:

* Programming skills
* Academic performance
* Number of projects
* Major / field of study
* Gender
* Age

Based on these features, the model predicts possible careers such as:

* Artificial Intelligence
* Data Science
* Software Development
* Web Development
* Cyber Security
* Cloud Engineering

---

# 🧠 Machine Learning Workflow

1️⃣ Data Collection

2️⃣ Data Cleaning & Preprocessing

3️⃣ Feature Encoding

4️⃣ Feature Scaling

5️⃣ Model Training

6️⃣ Hyperparameter Tuning

7️⃣ Model Evaluation

8️⃣ Streamlit Web App Deployment

---

# 📊 Dataset

The dataset contains **180 student records** with the following features:

| Feature  | Description                  |
| -------- | ---------------------------- |
| Age      | Student age                  |
| GPA      | Academic score               |
| Python   | Python skill level           |
| SQL      | SQL skill level              |
| Java     | Java skill level             |
| Projects | Number of completed projects |
| Major    | Student major                |
| Gender   | Student gender               |
| Career   | Target career                |

---

# 🤖 Machine Learning Model

Algorithm used:

**Random Forest Classifier**

Why Random Forest?

* Handles small datasets well
* Reduces overfitting
* Works well with mixed features
* Good performance with minimal tuning

---

# 📈 Model Performance

| Metric                 | Score |
| ---------------------- | ----- |
| Cross Validation Score | ~0.85 |

---

# 🖥 Streamlit Web Application

The project includes a **Streamlit UI** where users can enter student details and receive career predictions instantly.

### Input Fields

* Age
* GPA
* Python Skill
* SQL Skill
* Java Skill
* Number of Projects
* Major
* Gender

### Output

Predicted career recommendation.

---

# 📂 Project Structure

```
student-career-prediction
│
├── models
│   ├── career_model.pkl
│   ├── scaler.pkl
│   ├── feature_encoders.pkl
│   ├── target_encoder.pkl
│   └── model_columns.pkl
│
├── app.py
├── train_model.py
├── cs_students.csv
├── cs.ipynb
├── report.html
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

Clone the repository:

```
git clone(https://github.com/ramvilas273/student_career_prediction/tree/main)
cd student-career-prediction
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the Streamlit app:

```
streamlit run app.py
```

The app will open in your browser.

---

# 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Streamlit
* Pickle
* Git & GitHub

---

# 📌 Future Improvements

* Add larger dataset
* Improve class imbalance handling
* Try advanced models (XGBoost, LightGBM)
* Add career guidance suggestions
* Deploy on cloud

---

# 👨‍💻 Author

**Ram Vilas**

B.Tech Information Technology
Aspiring Data Scientist

GitHub: [https://github.com/ramvilas273]

---

# ⭐ If you like this project

Give it a ⭐ on GitHub!





