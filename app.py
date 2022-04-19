#install openpyxl, gunicorn, dash, plotly
from flask import Flask, jsonify
import joblib
from flask import Flask, render_template, request, url_for
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from waitress import serve

app = Flask(__name__)

data = pd.read_excel('data_training.xlsx')

data['Subject'].unique()

le = LabelEncoder()#manggil fungsinya
le.fit(list(data['Subject'].values))#ngambil value kolom subject 
data['Subject'] = le.transform(list(data['Subject']))#ngerubah isi kolom subject ke int
    
data['Judul']=data['Judul'].apply(lambda i:i.lower())
data['Judul'].str.split() #tempat nyeplit

token=data['Judul'].str.split() #deklarasiin tempat hasil split jadi token

stop = list(stopwords.words('indonesian'))

def rubah(data):
    token=data.split()#nunjukin data pada df pas displit
    a = ' '.join([word for word in token if (word  not in stop)&(word.isalpha())])#isalpha supaya gak ribet jadi satu df aplhabet semua yang dipake
    return a    

t = data['Judul'].apply(rubah)

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(data['Judul'])#merubah text jadi numerik dari df judul

@app.route('/')
def form():
    return render_template('form_submit.html')

@app.route('/predict', methods=['POST'])
def predict():
	s = request.form['judul']
	text = []
	text.append(s)
	text[0] = text[0].lower()
	arr = (vectorizer.transform(text))
	clf = joblib.load('best.pkl')
	prediction = (clf.predict(arr))
	prediction = le.inverse_transform(prediction)[0]
	return jsonify({'Subject adalah ': (prediction)})

if __name__ == '__main__':
	clf = joblib.load('best.pkl')
	serve(app, host='0.0.0.0', port=8080)#soalnya port python gak bisa lebih dari 60 detik

