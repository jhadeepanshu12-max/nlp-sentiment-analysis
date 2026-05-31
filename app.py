# app.py

from flask import Flask, render_template_string, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Flask app
app = Flask(__name__)

# -------------------------------
# Step 1: Training Data
# -------------------------------
data = {
    'text': [
        'I love this movie',
        'This movie is terrible',
        'Amazing storyline',
        'Worst film ever',
        'I really enjoyed it',
        'It was boring and bad',
        'I hate this',
        'What a wonderful experience',
        'I am very happy',
        'This is so disappointing',
        'Today was a good day'
    ],
    'sentiment': [
        'positive', 'negative', 'positive', 'negative',
        'positive', 'negative', 'negative', 'positive', 'positive', 'negative' , 'positive'
    ]
}

# -------------------------------
# Step 2: Train Model
# -------------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['sentiment']

model = MultinomialNB()
model.fit(X, y)

# -------------------------------
# Step 3: HTML Template (Inline)
# -------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Sentiment Analysis App</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(to right, #74ebd5, #ACB6E5);
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 0 20px rgba(0,0,0,0.2);
            width: 400px;
            text-align: center;
        }
        textarea {
            width: 90%;
            height: 100px;
            border-radius: 10px;
            border: 1px solid #ccc;
            padding: 10px;
            resize: none;
            font-size: 16px;
        }
        button {
            margin-top: 15px;
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            background: #007bff;
            color: white;
            font-size: 16px;
            cursor: pointer;
        }
        button:hover {
            background: #0056b3;
        }
        .result {
            margin-top: 20px;
            font-size: 18px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>🧠 Sentiment Analysis Tool</h2>
        <form method="POST" action="/">
            <textarea name="text" placeholder="Enter your text here..." required></textarea><br>
            <button type="submit">Analyze</button>
        </form>
        {% if prediction %}
        <div class="result">
            <p><b>Text:</b> {{ text }}</p>
            <p><b>Sentiment:</b> {{ prediction }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    text = ""
    if request.method == 'POST':
        text = request.form['text']
        text_vec = vectorizer.transform([text])
        prediction = model.predict(text_vec)[0]
    return render_template_string(HTML_TEMPLATE, prediction=prediction, text=text)

# -------------------------------
# Step 5: Run App
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)
