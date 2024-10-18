from flask import Flask, request, render_template, redirect, url_for, send_file
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import os
import re
import spacy
from nltk.corpus import stopwords

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your own secret key

# Load the spaCy model for lemmatization
nlp = spacy.load('en_core_web_sm')

# Define stop words using NLTK
stop_words_nltk = set(stopwords.words('english'))

# Sentiment Analysis Function using TextBlob
def analyze_sentiment(review):
    analysis = TextBlob(review)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Emotion Assignment Function
def assign_emotion(sentiment):
    if sentiment == 'Positive':
        return 'Joyful😄'
    elif sentiment == 'Negative':
        return 'Angry😠'
    else:
        return 'Calm😌'

# Define a function for text preprocessing
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = ' '.join(word for word in text.split() if word not in stop_words_nltk)  # Remove stopwords
    doc = nlp(text)  # Apply lemmatization
    text = ' '.join(token.lemma_ for token in doc)
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None  # Initialize results to None

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return f"Error processing the file: {str(e)}"

        # Check for necessary columns
        if 'reviews' not in df.columns:
            return "Error: The dataset must contain 'reviews' column."

        # Data cleaning and preprocessing
        df = df.drop_duplicates()  # Remove duplicate rows
        df = df.dropna(subset=['reviews'])  # Remove rows with missing reviews
        df['reviews'] = df['reviews'].apply(preprocess_text)  # Apply text preprocessing

        # Optional: Check if 'true_sentiment' column is available for accuracy calculation
        true_sentiment_column = 'true_sentiment'  # Change this if your true sentiment column has a different name
        has_true_sentiment = true_sentiment_column in df.columns

        reviews = df['reviews'].tolist()
        sentiments = [analyze_sentiment(review) for review in reviews]
        emotions = [assign_emotion(sentiment) for sentiment in sentiments]

        # Add the sentiment and emotion results to the dataframe
        df['Sentiment'] = sentiments
        df['Emotion'] = emotions

        # Save the analyzed dataset to a CSV file
        analyzed_file_path = os.path.join('static', 'analyzed_reviews.csv')
        df.to_csv(analyzed_file_path, index=False)

        # Count the number of positive, negative, and neutral reviews
        sentiment_counts = {
            'Positive': sentiments.count('Positive'),
            'Negative': sentiments.count('Negative'),
            'Neutral': sentiments.count('Neutral')
        }

        # Calculate total count
        total_count = len(reviews)
        total_positive = sentiment_counts['Positive']
        total_negative = sentiment_counts['Negative']
        total_neutral = sentiment_counts['Neutral']

        # Calculate top positive and negative reviews (optional)
        top_positive = df[df['Sentiment'] == 'Positive']['reviews'].mode().iloc[0] if total_positive > 0 else "None"
        top_negative = df[df['Sentiment'] == 'Negative']['reviews'].mode().iloc[0] if total_negative > 0 else "None"

        # Calculate accuracy and error rate if true sentiment is available
        accuracy = 0
        error_rate = 0
        if has_true_sentiment:
            correct_predictions = (df['Sentiment'] == df[true_sentiment_column]).sum()
            accuracy = correct_predictions / total_count
            error_rate = 1 - accuracy
        else:
            accuracy = None
            error_rate = None

        # Plot the sentiment distribution
        plt.figure(figsize=(8, 6))
        plt.bar(sentiment_counts.keys(), sentiment_counts.values(), color=['#669963', '#fb5462', '#ffca01'])
        plt.xlabel('Sentiment')
        plt.ylabel('Number of Reviews')
        plt.title('Sentiment Analysis Distribution')
        
        # Save the plot as an image file
        chart_path = os.path.join('static', 'sentiment_chart.png')
        plt.savefig(chart_path)
        plt.close()

        # Prepare the reviews data for the template
        # Include emotions in the tuple
        results_reviews = list(zip(reviews, sentiments, emotions))

        results = {
            'reviews': results_reviews,
            'chart': chart_path,
            'analyzed_file': analyzed_file_path,
            'total_count': total_count,
            'total_positive': total_positive,
            'total_negative': total_negative,
            'total_neutral': total_neutral,
            'top_positive': top_positive,
            'top_negative': top_negative,
            'accuracy': accuracy,
            'error_rate': error_rate,
        }

    return render_template('index.html', results=results)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/download')
def download_file():
    path = request.args.get('path')
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    else:
        return "File not found", 404
    
    

if __name__ == '__main__':
    app.run(debug=True)
