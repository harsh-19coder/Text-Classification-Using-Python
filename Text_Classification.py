# Import necessary libraries
import pandas as pd
import zipfile
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import precision_score, recall_score, f1_score

# Download the zipped CSV file from the internet
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
filename = 'dataset.zip'
urllib.request.urlretrieve(url, filename)

# Unzip the CSV file
with zipfile.ZipFile(filename, 'r') as zipf:
    zipf.extractall()
      
# Load your dataset from the unzipped CSV file
df = pd.read_csv('SMSSpamCollection', sep='\t', names=["label", "sms_text"])
# Preprocess your data
X = df['sms_text']
y = df['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert text data into numerical data
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Define the models
nb = MultinomialNB()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
svm_model = svm.SVC()

# Train the models
nb.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

# Make predictions
nb_pred = nb.predict(X_test)
dt_pred = dt.predict(X_test)
rf_pred = rf.predict(X_test)
svm_pred = svm_model.predict(X_test)

# Evaluate the models for both "ham" and "spam"
for label in ['ham', 'spam']:
    print(f"Evaluation for label: {label}")
    y_test_label = (y_test == label)
    print("\n")
    # Evaluate the model of NB
    nb_pred_label = (nb_pred == label)
    print("NB Precision: ", round(precision_score(y_test_label, nb_pred_label),2))
    print("NB Recall: ", round(recall_score(y_test_label, nb_pred_label),2))
    print("NB F1 Score: ", round(f1_score(y_test_label, nb_pred_label),2))
    print("\n")
    
    # Evaluate the model of Decision Tree
    dt_pred_label = (dt_pred == label)
    print("DT Precision: ", round(precision_score(y_test_label, dt_pred_label),2))
    print("DT Recall: ", round(recall_score(y_test_label, dt_pred_label),2))
    print("DT F1 Score: ", round(f1_score(y_test_label, dt_pred_label),2))
    print("\n")
    
    # Evaluate the model of Random Forest
    rf_pred_label = (rf_pred == label)
    print("RF Precision: ", round(precision_score(y_test_label, rf_pred_label),2))
    print("RF Recall: ", round(recall_score(y_test_label, rf_pred_label),2))
    print("RF F1 Score: ", round(f1_score(y_test_label, rf_pred_label),2))
    print("\n")

    # Evaluate the model of SVM
    svm_pred_label = (svm_pred == label)
    print("SVM Precision: ", round(precision_score(y_test_label, svm_pred_label),2))
    print("SVM Recall: ", round(recall_score(y_test_label, svm_pred_label),2))
    print("SVM F1 Score: ", round(f1_score(y_test_label, svm_pred_label),2))
    print("\n\n\n\n")
    
