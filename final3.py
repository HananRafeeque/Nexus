import os
from flask import Flask, render_template, request, redirect, send_file, url_for, session, flash
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin, current_user
from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import pandas as pd
import matplotlib
from wtforms import Form
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
from flask_wtf.csrf import CSRFProtect
from forms import RegistrationForm, UploadForm
from forms import LoginForm
from flask_login import current_user


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


app = Flask(__name__)
app.secret_key = 'N3X65'  #  The secret key

csrf = CSRFProtect(app)  # Enabled CSRF protection
login_manager = LoginManager(app)  # Initialize LoginManager
login_manager.login_view = 'login'  # Set login view for unauthorized access redirection

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
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    email = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(200), nullable=False)

    # User loader
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create database tables if they don't exist
with app.app_context():
    db.create_all()


# Function to predict churn based on sentiment
def predict_churn(sentiment):
    if sentiment == 'Negative':
        return 'Non-Retained'  # Churn
    else:
        return 'Retained'  # Not Churn



    


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

SVM_CM_PATH = os.path.join('static', 'svm_confusion_matrix.png')
RF_CM_PATH = os.path.join('static', 'rf_confusion_matrix.png')
NB_CM_PATH = os.path.join('static', 'nb_confusion_matrix.png')

# Function to train and evaluate models
def train_and_evaluate_models(df):
    text = df['reviews'].values
    labels = df['rating'].values
    
    # Remove NaN values from text data
    text = [str(doc) if not pd.isnull(doc) else '' for doc in text]
    
    # Split the dataset into training and testing sets
    text_train, text_test, labels_train, labels_test = train_test_split(text, labels, test_size=0.2, random_state=42)
    


    # Random Forest Model
    rf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = rf_vectorizer.fit_transform(text_train)
    X_test_tfidf = rf_vectorizer.transform(text_test)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_tfidf, labels_train)
    rf_pred = rf_model.predict(X_test_tfidf)
    rf_accuracy = accuracy_score(labels_test, rf_pred)

    # Generate and save Random Forest confusion matrix
    conf_matrix_rf = confusion_matrix(labels_test, rf_pred)
    disp_rf = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_rf)
    disp_rf.plot(cmap="Purples")
    plt.title("Random Forest Confusion Matrix")
    plt.savefig(RF_CM_PATH)
    plt.close()
    

    # Return model performances
    return {
        # 'svm_accuracy': svm_accuracy,
        'rf_accuracy': rf_accuracy,
        # 'nb_accuracy': nb_accuracy,
        # 'svm_report': classification_report(labels_test, svm_predictions),
        'rf_report': classification_report(labels_test, rf_pred),
        # 'nb_report': classification_report(labels_test, nb_predictions),

        'svm_cm_path': SVM_CM_PATH,
        'rf_cm_path': RF_CM_PATH,
        'nb_cm_path': NB_CM_PATH,
    }
    


#code for clear data
@app.route('/clear_data', methods=['POST'])
@login_required
def clear_data():
    # Delete the analyzed data file if it exists
    if os.path.exists(ANALYZED_DATA_PATH):
        os.remove(ANALYZED_DATA_PATH)
        flash("Analyzed data cleared successfully.", "success")

    # Delete the chart image if it exists
    if os.path.exists(CHART_PATH):
        os.remove(CHART_PATH)
        flash("Chart cleared successfully.", "success")

    # Redirect back to the dashboard
    return redirect(url_for('home'))



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
        
        flash('Registration  successful! You can log-in now.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html',form=form)


def clear_analyzed_data():
    """Removes previous analyzed data and chart files if they exist."""
    if os.path.exists(ANALYZED_DATA_PATH):
        os.remove(ANALYZED_DATA_PATH)
    if os.path.exists(CHART_PATH):
        os.remove(CHART_PATH)



# Route for the login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    clear_analyzed_data()
    form = LoginForm()
    if form.validate_on_submit():  # to use Flask-WTF's form validation and CSRF protection
        username = form.username.data  # to access form data via Flask-WTF
        password = form.password.data  

        # Check if user exists
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)  
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Login failed. Please check your username and password.', 'error')
            return redirect(url_for('login'))
    
    return render_template('login.html', form=form)


def churn_prediction(sentiment):
    # Provide a more accurate churn prediction based on sentiment
    if sentiment > 0:
        return 'Retain'
    elif sentiment < 0:
        return 'Non Retain'
    else:
        return 'Maybe Retain or Non Retain' 

@app.route('/home', methods=['GET', 'POST'])
@login_required
def home():

    
    form = UploadForm()
    sentiment_result = None  # Initialize sentiment_result
    error_message = None  # Initialize error_message
    

    if form.validate_on_submit():  # Ensure form validation and CSRF protection
        file = request.files.get('file')

        # Clear button check
        if 'clear' in request.form:
            # Remove analyzed dataset and chart files if they exist
            if os.path.exists(ANALYZED_DATA_PATH):
                os.remove(ANALYZED_DATA_PATH)
                print("Removed analyzed_reviews.csv")
            if os.path.exists(CHART_PATH):
                os.remove(CHART_PATH)
                print("Removed sentiment_chart.png")
            flash('Data cleared successfully.', 'success')
            return render_template('home.html', form=form, chart_exists=False, username=current_user.username)

        # File upload processing
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
            return render_template('home.html', form=form)

        if 'reviews' not in df.columns:
            flash("The dataset must contain a 'reviews' column.", 'error')
            return render_template('home.html', form=form)
        


        # Preprocessing and sentiment analysis
        df = df.drop_duplicates()
        df = df.dropna(subset=['reviews'])
        df['reviews'] = df['reviews'].apply(preprocess_text)
        df['Sentiment'] = df['reviews'].apply(analyze_sentiment)
        df['Emotion'] = df['Sentiment'].apply(assign_emotion)

        # Use the identified rating column
        df['rating'] = df[rating_column]
        df.to_csv(ANALYZED_DATA_PATH, index=False)

        # Train and evaluate models, then print results in the terminal
        model_results = train_and_evaluate_models(df)

        print("Random Forest Model Accuracy:", model_results['rf_accuracy'])
        print("Random Forest Classification Report:\n", model_results['rf_report'])

        return redirect(url_for('dashboard'))
    
    # Review analysis logic
    if 'review' in request.form:
        review = request.form['review']
        blob = TextBlob(review)
        sentiment = blob.sentiment.polarity

        if sentiment > 0:
            sentiment_result = 'Positive'
        elif sentiment < 0:
            sentiment_result = 'Negative'
        else:
            sentiment_result = 'Neutral'
        

        return render_template('home.html', form=form, sentiment_result=sentiment_result, username=current_user.username)

    return render_template('home.html', form=form, chart_exists=False, username=current_user.username)


    



# Route for the about page
@app.route('/about')
@login_required
def about():
    return render_template('about.html')

# Route for the team page
@app.route('/team')
@login_required
def team():
    return render_template('team.html')

@app.route('/dashboard')
@login_required
def dashboard():
    if os.path.exists(ANALYZED_DATA_PATH):
        df = pd.read_csv(ANALYZED_DATA_PATH)

        # Prepare the reviews for display
        reviews = df.to_dict(orient='records')  # Convert DataFrame to a list of dictionaries

        # Count the total number of reviews, positive reviews, and negative reviews
        total_reviews = len(df)
        positive_reviews = len(df[df['Sentiment'] == 'Positive'])
        negative_reviews = len(df[df['Sentiment'] == 'Negative'])
        neutral_reviews = len(df[df['Sentiment'] == 'Neutral'])

        # Find top positive and top negative reviews if they exist
        top_positive = df[df['Sentiment'] == 'Positive']['reviews'].mode().iloc[0] if positive_reviews > 0 else "None"
        top_negative = df[df['Sentiment'] == 'Negative']['reviews'].mode().iloc[0] if negative_reviews > 0 else "None"
        for review in reviews:
            review['Churn'] = 'Non-retain' if review['Sentiment'] == 'Negative' else 'Retain'


        # Check if there's a chart generated
        chart_exists = os.path.exists(CHART_PATH)

        return render_template('dashboard.html', chart_exists=chart_exists, reviews=reviews,
                               total_reviews=total_reviews, positive_reviews=positive_reviews,
                               negative_reviews=negative_reviews, neutral_reviews=neutral_reviews,
                               top_positive=top_positive, top_negative=top_negative)
    else:
        return render_template('dashboard.html', chart_exists=False, reviews=[], total_reviews=0,
                               positive_reviews=0, negative_reviews=0, neutral_reviews=0,
                               top_positive="None", top_negative="None")


@app.route('/download')
@login_required
def download_file():
    if os.path.exists(ANALYZED_DATA_PATH):
        return send_file(ANALYZED_DATA_PATH, as_attachment=True)
    return "No file to download."

@app.route('/download_chart')
@login_required
def download_chart():
    if os.path.exists(CHART_PATH):
        return send_file(CHART_PATH, as_attachment=True)
    return "No chart to download."
    

# Route for logging out
@app.route('/logout')
def logout():
    logout_user()  # Log the user out using Flask-Login.
    flash('You have been logged-out.', 'success')  # Flash a success message
    return redirect(url_for('login'))


        

if __name__ == '__main__':
    app.run(debug=True)



