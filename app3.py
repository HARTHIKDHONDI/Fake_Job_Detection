from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
scaler= StandardScaler()
## init Flask App
app3 = Flask(__name__)

# Load Pickle model
loaded_model = pickle.load(open('C:/Users/dhondik/OneDrive/Desktop/FAKE JOB LISTING DETECTION/job1.pkl','rb'))

# We need to fit the TFIDF VEctorizer
df = pd.read_csv('C:/Users/dhondik/OneDrive/Desktop/FAKE JOB LISTING DETECTION/fake_job_postings.csv')

df['employment_type'].fillna(' ',inplace = True)
df['required_experience'].fillna(' ',inplace = True)
df['required_education'].fillna(' ',inplace = True)
df['function'].fillna(' ',inplace = True)


le = LabelEncoder()
df['employment_type'] = le.fit_transform(df['employment_type'])
df['required_experience'] = le.fit_transform(df['required_experience'])
df['required_education'] = le.fit_transform(df['required_education'])
df['function'] = le.fit_transform(df['function'])
df = df.reset_index()
X = df[['telecommuting', 'has_company_logo','has_questions', 'employment_type','required_experience', 'required_education','function']]
Y = df['fraudulent']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,random_state=0,test_size=.30)




def fake_job_det(telecommuting,has_company_logo,has_questions,employment_type,required_experience,required_education,function):
    input_data = [telecommuting,has_company_logo,has_questions,employment_type,required_experience,required_education,function]
    print(input_data)
    
    prediction = loaded_model.predict([input_data])
    return prediction
# Defining the site route
@app3.route('/')
def home():
    return render_template('index3.html')

@app3.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        telecommuting = request.form['telecommuting']
        has_company_logo=request.form['has_company_logo']
        has_questions=request.form['has_questions']
        employment_type=request.form['employment_type']
        required_experience=request.form['required_experience']
        required_education=request.form['required_education']
        function=request.form['function']
        pred = fake_job_det(telecommuting,has_company_logo,has_questions,employment_type,required_experience,required_education,function)
        print(pred)
        return render_template('index3.html', prediction=pred)
    else:
        return render_template('index3.html', prediction="Something went wrong")

if __name__ == '__main__':
    app3.run(debug=True)