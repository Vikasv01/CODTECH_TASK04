#  Spam Detection Classifier | CodTech Internship - Task 04

This project is part of the CodTech Python internship. The objective is to build a **machine learning model** using **Scikit-learn** to classify SMS messages as **Spam** or **Not Spam (Ham)** using natural language processing techniques.

---

##  Task Objective

> **Task 04 – Machine Learning Model Implementation**  
> Create a predictive model using Scikit-learn to classify or predict outcomes from a dataset (e.g., spam email detection).

---

##  Project Structure

<pre>TASK04_CODTECH/
├── spam_classifier.py # Main Python script
├── spam.csv # Dataset (from Kaggle)
├── requirements.txt # Project dependencies
└── README.md # Project documentation</pre>

---

###  Requirements

Install the required libraries using:
pip install -r requirements.txt

requirements.txt

pandas
numpy
scikit-learn
matplotlib

---

### Dataset Used
Source: SMS Spam Collection Dataset - Kaggle

Format: CSV with two columns:

v1 → Label (ham or spam)

v2 → Message content

---

### How the Model Works

1. Data Cleaning – Extract required columns, rename, and map labels.

2. Text Vectorization – Convert text into numerical form using CountVectorizer.

3. Model Training – Train a Multinomial Naive Bayes classifier.

4. Model Evaluation – Evaluate accuracy, print classification report, and plot confusion matrix.

---

### How to Run

1. Ensure spam.csv is in the project directory.

2.  Run the Python script:
   
   python spam_classifier.py

---

### Output Example

Accuracy: 0.984
Precision, Recall, F1-score...

Confusion Matrix plotted in a separate window.

---

### Sample Confusion Matrix

The script also generates a heatmap of the confusion matrix showing predictions vs. actual labels.

---

### Future Enhancements

1. Use TF-IDF vectorization

2. Try different ML algorithms (Logistic Regression, SVM)

3. Apply deep learning using LSTM for advanced performance

4. Build a Flask web app around the model

---
###
