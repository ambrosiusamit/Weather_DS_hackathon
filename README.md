# Weather_DS_hackathon
A data science project developed for a hackathon with the goal of building a predictive model for weather classification using machine learning. This project explores various aspects of data preprocessing, feature engineering, classification modeling, and performance evaluation.

#Project Overview
Weather plays a vital role in our daily lives, and accurate classification of weather types can significantly benefit various sectors, including agriculture, aviation, and logistics. This project aims to classify weather into categories such as Clear, Cloudy, Foggy, Rainy, Snowy, etc., using machine learning techniques on meteorological data.


# Dataset
The dataset consists of labeled weather data with features such as:

Temperature (°C)
Humidity (%)
Wind Speed (km/h)
Visibility (km)
Pressure (hPa)
Dew Point (°C)
Precipitation (mm)
Target Variable: Weather Condition (Categorical)

#Files
train.csv – training data with weather labels
test.csv – test data without labels (to be predicted)


#Exploratory Data Analysis (EDA)
Distribution of numeric features
Missing value analysis and imputation
Correlation heatmaps
Class distribution
Feature importance visualization
EDA was performed using Python libraries like pandas, matplotlib, and seaborn.


# Model Building
Several machine learning models were experimented with:
Logistic Regression
Random Forest
XGBoost
LightGBM
Support Vector Machine
K-Nearest Neighbors
Performance was evaluated using accuracy, F1-score, precision, and recall. Hyperparameter tuning was done using GridSearchCV and RandomizedSearchCV.

#Ensemble Modeling
An ensemble of the top-performing models was created using:
Voting Classifier (Hard & Soft)
StackingClassifier
The ensemble approach significantly improved overall classification accuracy.

#Evaluation Metrics
Accuracy
Precision
Recall
F1-Score
Confusion Matrix
The final model achieved high accuracy and balanced class-wise performance.

# Requirements
To install the dependencies, run:
bash
Copy
Edit

pip install -r requirements.txt
Key libraries:
pandas
numpy
scikit-learn
xgboost
lightgbm
matplotlib
seaborn

  #How to Run

Clone the repository:
bash
Copy
Edit
git clone https://github.com/ambrosiusamit/Weather_DS_hackathon.git
cd Weather_DS_hackathon

Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt

Open the notebooks in JupyterLab or VS Code and run them in order:
EDA.ipynb
Model_Building.ipynb
Ensemble_Modeling.ipynb
Generate final predictions and save to submission.csv.
Also replace API_KEY with you real key

# Highlights
Strong focus on data preprocessing and class balancing
Use of both simple and advanced ML models
Ensemble modeling boosts performance
Clean and modular Jupyter notebooks

# Authors
Amit Ambrosius

Paras lookcoates
