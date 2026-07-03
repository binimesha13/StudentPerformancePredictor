# Student Score Predictor + Goal Optimizer

A machine learning web app that predicts a student's exam score based on demographic and academic factors — and goes a step further with a **Goal Optimizer**: tell it your target score, and it tells you exactly which factor to improve and by how much to get there.

## Live Demo
 https://binimesha13-studentperformancepredictor-student-dfechu.streamlit.app/

## Features
- Predicts Math, Reading, or Writing score using student demographic and academic data
- Compares **Linear Regression** vs **Random Forest Regressor**, and auto-selects the better performing model based on R² and MAE
- **Goal Optimizer**: reverse-engineers the model's prediction to recommend which factor to improve, and by how much, to hit a target score
- Visualizes feature importance to explain what drives the predicted score
- Interactive UI built with Streamlit

## Tech Stack
- **Python**
- **Scikit-learn** – Linear Regression, Random Forest, model evaluation
- **Pandas / NumPy** – data processing
- **Matplotlib** – feature importance visualization
- **Streamlit** – web app interface

## How to Run Locally

```bash
git clone https://github.com/binimesha13/StudentPerformancePredictor.git
cd StudentPerformancePredictor
pip install -r requirements.txt
streamlit run student.py
```

## Project Structure

```
StudentPerformancePredictor/
│
├── student.py           # Main application code
├── requirements.txt    # Python dependencies
├── student_data.csv    # Dataset (Students Performance in Exams)
└── README.md
```

## Model Performance
- Linear Regression achieved an R² score of ~0.88 on test data
- Random Forest Regressor used as a comparison model for non-linear relationships

## Future Scope
- Add more ML models (XGBoost, Gradient Boosting) to the comparison
- Deploy with a database to track student progress over time
- Add model explainability using SHAP values

## Author
Binimesha Naisargika-https://www.linkedin.com/in/binimesha-naisargika-26b176291/