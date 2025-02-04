from flask import Flask, request, render_template
from textblob import TextBlob

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    if request.method == 'POST':
        comment = request.form['comment']
        
        blob = TextBlob(comment)
        sentiment = blob.sentiment.polarity  
        
        if sentiment > 0:
            sentiment_label = "Positive"
        elif sentiment < 0:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        
        return render_template('index.html', sentiment=sentiment_label)

if __name__ == '__main__':
    app.run(debug=True)