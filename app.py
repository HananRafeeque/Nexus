from flask import Flask, request, render_template, redirect, url_for, send_file
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Sentiment Analysis Function using TextBlob
def analyze_sentiment(review):
    analysis = TextBlob(review)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        df = pd.read_csv(file)

        # Assuming 'reviews' is the column with reviews
        reviews = df['reviews'].tolist()
        sentiments = []

        for review in reviews:
            sentiment = analyze_sentiment(review)
            sentiments.append(sentiment)

        # Add the sentiment results to the dataframe
        df['Sentiment'] = sentiments

        # Save the analyzed dataset to a CSV file
        analyzed_file_path = os.path.join('static', 'analyzed_reviews.csv')
        df.to_csv(analyzed_file_path, index=False)

        # Count the number of positive, negative, and neutral reviews
        sentiment_counts = {
            'Positive': sentiments.count('Positive'),
            'Negative': sentiments.count('Negative'),
            'Neutral': sentiments.count('Neutral')
        }

        # Plot the sentiment distribution
        plt.figure(figsize=(8, 6))
        plt.bar(sentiment_counts.keys(), sentiment_counts.values(), color=['#b5196a', '#feb800', '#17552e'])
        plt.xlabel('Sentiment')
        plt.ylabel('Number of Reviews')``
        plt.title('Sentiment Analysis Distribution')
        
        # Save the plot as an image file
        chart_path = os.path.join('static', 'sentiment_chart.png')
        plt.savefig(chart_path)
        plt.close()

        results = {
            'reviews': zip(reviews, sentiments),
            'chart': chart_path,
            'analyzed_file': analyzed_file_path
        }

        return render_template('index.html', results=results)
    
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/download')
def download_file():
    path = request.args.get('path')
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
