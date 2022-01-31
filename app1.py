from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

## init Flask App
app1 = Flask(__name__)

# Load Pickle model
loaded_model = pickle.load(open('C:/Users/dhondik/OneDrive/Desktop/FAKE JOB LISTING DETECTION/job.pkl','rb'))

# We need to fit the TFIDF VEctorizer
df = pd.read_csv('C:/Users/dhondik/OneDrive/Desktop/FAKE JOB LISTING DETECTION/fake_job_postings.csv')

df['description'].fillna(' ',inplace = True)
x = df['description']
y = df['fraudulent']

X_train, X_test, y_train, y_test = train_test_split(x,y,random_state=0,test_size=.30)
tfidf_vectorizer = TfidfVectorizer(stop_words="english", min_df=0.05, max_df=0.9)

def fake_job_det(description):
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    tfidf_test = tfidf_vectorizer.transform(X_test)
    input_data = [description]
    vectorized_input_data = tfidf_vectorizer.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction
# Defining the site route
@app1.route('/')
def home():
    return render_template('index1.html')

@app1.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        description = request.form['description']
        pred = fake_job_det(description)
        print(pred)
        return render_template('index1.html', prediction=pred)
    else:
        return render_template('index1.html', prediction="Something went wrong")

if __name__ == '__main__':
    app1.run(debug=True)