# - PROJECT
#     - css
#         - styles.css
#     -data
#         -xAPI-Edu-Data.csv 
#     - index.html
#     -model_training

# import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from flask import Flask, render_template, request


# Load the dataset
data = pd.read_csv('data/xAPI-Edu-Data.csv')
# Separate features and target variable
features = data.drop('Class', axis=1)
target = data['Class']

# Encode categorical variables
label_encoder = LabelEncoder()
features_encoded = features.apply(label_encoder.fit_transform)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_encoded, target, test_size=0.2, random_state=42)

# Initialize and train the random forest classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'trained_model.pkl')

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the evaluation metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')


# Model Evaluation
def evaluate_model(model, X, y):
    # Make predictions
    y_pred = model.predict(X)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')

    return accuracy, precision, recall, f1


# Load the trained model
def load_model(file_path):
    model = joblib.load(file_path)
    return model

# Initialize the Flask app
app = Flask(__name__, template_folder='Project', static_folder='Project/css')


# Define routes and HTML templates
@app.route('/')
def index():
    return render_template('Project/templates/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Handle user input and make predictions here
    # ...

    prediction = "Sample Prediction" 
    return render_template('result.html', prediction=prediction)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=8000)

