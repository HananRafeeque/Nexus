import os
from flask import Flask, render_template, request, redirect, send_file, url_for, session, flash 
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import re
import spacy
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from forms import RegistrationForm 
from forms import LoginForm

app = Flask(__name__)
app.secret_key = 'N3X65'  # The secret key

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# File paths
ANALYZED_DATA_PATH = os.path.join('static', 'analyzed_reviews.csv')
CHART_PATH = os.path.join('static', 'sentiment_chart.png')

# Loading the spaCy model for lemmatization
nlp = spacy.load('en_core_web_sm')

# Define stop words using NLTK
stop_words_nltk = set(stopwords.words('english'))

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    email = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(200), nullable=False)

# Create database tables if they don't exist
with app.app_context():
    db.create_all()

# Sentiment Analysis Function using TextBlob
def analyze_sentiment(review):
    analysis = TextBlob(review)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Define a function to assign emotions based on sentiment
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

# Remove previous data files at the start of the application
if os.path.exists(ANALYZED_DATA_PATH):
    os.remove(ANALYZED_DATA_PATH)
if os.path.exists(CHART_PATH):
    os.remove(CHART_PATH)

# Function to train and evaluate models
def train_and_evaluate_models(df):
    text = df['reviews'].values
    labels = df['rating'].values
    
    # Remove NaN values from text data
    text = [str(doc) if not pd.isnull(doc) else '' for doc in text]
    
    # Split the dataset into training and testing sets
    text_train, text_test, labels_train, labels_test = train_test_split(text, labels, test_size=0.2, random_state=42)
    
    # SVM Model
    svm_vectorizer = CountVectorizer()
    svm_features_train = svm_vectorizer.fit_transform(text_train)
    svm_features_test = svm_vectorizer.transform(text_test)
    svm = SVC(kernel='linear')
    svm.fit(svm_features_train, labels_train)
    svm_predictions = svm.predict(svm_features_test)
    svm_accuracy = accuracy_score(labels_test, svm_predictions)

    # Random Forest Model
    rf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = rf_vectorizer.fit_transform(text_train)
    X_test_tfidf = rf_vectorizer.transform(text_test)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_tfidf, labels_train)
    rf_pred = rf_model.predict(X_test_tfidf)
    rf_accuracy = accuracy_score(labels_test, rf_pred)

    # Naive Bayes Model
    nb_vectorizer = CountVectorizer()
    nb_features_train = nb_vectorizer.fit_transform(text_train)
    nb_features_test = nb_vectorizer.transform(text_test)
    nb = MultinomialNB()
    nb.fit(nb_features_train, labels_train)
    nb_predictions = nb.predict(nb_features_test)
    nb_accuracy = accuracy_score(labels_test, nb_predictions)

    # Return model performances
    return {
        'svm_accuracy': svm_accuracy,
        'rf_accuracy': rf_accuracy,
        'nb_accuracy': nb_accuracy,
        'svm_report': classification_report(labels_test, svm_predictions),
        'rf_report': classification_report(labels_test, rf_pred),
        'nb_report': classification_report(labels_test, nb_predictions),
    }

# Route for the homepage
@app.route('/')
def intro():
    return render_template('intro.html')

# Route for the registration page
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():  # Tohandle form validation and CSRF protection
        username = form.username.data
        email = form.email.data
        password = form.password.data
        
        # Check if the user already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists. Please choose a different one.', 'error')
            return redirect(url_for('register'))

        # Hash the password and save the new user to the database
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html' , form=form)

# Route for the login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():  # to use Flask-WTF's form validation and CSRF protection
        username = form.username.data  # to access form data via Flask-WTF
        password = form.password.data  

        # Check if user exists
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Login failed. Please check your username and password.', 'error')
            return redirect(url_for('login'))
    
    return render_template('login.html', form=form)

# Route for the home page after login

@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'clear' in request.form:
            if os.path.exists(ANALYZED_DATA_PATH):
                os.remove(ANALYZED_DATA_PATH)
                print("Removed analyzed_reviews.csv")
            if os.path.exists(CHART_PATH):
                os.remove(CHART_PATH)
                print("Removed sentiment_chart.png")
            return render_template('home.html', chart_exists=False)

        if 'file' not in request.files:
            flash('No file selected.', 'error')
            return redirect(request.url)

        file = request.files['file']
        if not file:
            flash('No file selected.', 'error')
            return redirect(request.url)

        try:
            df = pd.read_csv(file)
            print(df.head())  # Check the DataFrame
        except Exception as e:
            flash(f"Error processing the file: {str(e)}", 'error')
            return redirect(request.url)

        # Dynamically find the rating column
        rating_column = None
        possible_rating_columns = ['rating', 'score', 'stars']
        for col in df.columns:
            if col.lower() in possible_rating_columns:
                rating_column = col
                print(f"Detected rating column: {rating_column}")  # Debugging statement
                break

        if rating_column is None:
            flash("Could not find a rating column in the dataset.", 'error')
            return render_template('home.html')

        if 'reviews' not in df.columns:
            flash("The dataset must contain a 'reviews' column.", 'error')
            return render_template('home.html')

        # Preprocessing and sentiment analysis
        df = df.drop_duplicates()
        df = df.dropna(subset=['reviews'])
        df['reviews'] = df['reviews'].apply(preprocess_text)
        df['Sentiment'] = df['reviews'].apply(analyze_sentiment)
        df['Emotion'] = df['Sentiment'].apply(assign_emotion)

        # Use the identified rating column
        df['rating'] = df[rating_column]
        df.to_csv(ANALYZED_DATA_PATH, index=False)

        # Generate sentiment counts and chart as before...
        # ...
        return redirect(url_for('dashboard'))

    return render_template('home.html', chart_exists=False)

# Route for the about page
@app.route('/about')
def about():
    return render_template('about.html')

# Route for the team page
@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/dashboard')
def dashboard():
    if os.path.exists(ANALYZED_DATA_PATH):
        df = pd.read_csv(ANALYZED_DATA_PATH)

        # Prepare the reviews for display
        reviews = df.to_dict(orient='records')  # Convert DataFrame to a list of dictionaries

        # Check if there's a chart generated
        chart_exists = os.path.exists(CHART_PATH)

        return render_template('dashboard.html', chart_exists=chart_exists, reviews=reviews)
    else:
        return render_template('dashboard.html', chart_exists=False, reviews=[])

import os


@app.route('/download')
def download_file():
    if os.path.exists(ANALYZED_DATA_PATH):
        return send_file(ANALYZED_DATA_PATH, as_attachment=True)
    return "No file to download."

@app.route('/download_chart')
def download_chart():
    if os.path.exists(CHART_PATH):
        return send_file(CHART_PATH, as_attachment=True)
    return "No chart to download."
    

# Route for logging out
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('intro'))

if __name__ == '__main__':
    app.run(debug=True)