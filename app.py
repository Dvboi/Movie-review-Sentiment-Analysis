import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from flask import Flask,render_template,url_for,request

lemmatizer = WordNetLemmatizer()

def lemma(doc):
    ans = [lemmatizer.lemmatize(text,'v') for text in doc.split()]
    return ans
model = pickle.load(open("logistic_model.pkl","rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction',methods=['GET','POST'])
def prediction():
    if request.method=='POST':
        temp = request.form['message']
        final = model.predict([temp])
    return render_template('prediction.html',ans=final)

if __name__ == "__main__":
    app.run()