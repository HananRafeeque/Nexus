# import pandas as pd
# import re
# import spacy
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer

# # Ensure necessary NLTK data is available
# nltk.download('stopwords')

# # Load the spaCy model for lemmatization
# nlp = spacy.load('en_core_web_sm')

# # Define stop words using NLTK
# stop_words_nltk = set(stopwords.words('english'))

# # Define a function for text preprocessing
# def preprocess_text(text):
#     # Lowercasing
#     text = text.lower()
    
#     # Remove punctuation and numbers
#     text = re.sub(r'[^\w\s]', '', text)
#     text = re.sub(r'\d+', '', text)
    
#     # Remove stop words
#     text = ' '.join(word for word in text.split() if word not in stop_words_nltk)
    
#     # Lemmatization
#     doc = nlp(text)
#     text = ' '.join(token.lemma_ for token in doc)
    
#     return text

# # Define a function to convert ratings to sentiment
# def rating_to_sentiment(rating):
#     if rating > 3:
#         return 'positive'
#     elif rating < 3:
#         return 'negative'
#     else:
#         return 'neutral'

# # Load the dataset
# df = pd.read_csv('data.csv')

# # Display the actual column names to inspect
# print("Column Names:", df.columns)

# # Rename columns if necessary
# # df.rename(columns={'reviews': 'reviews.text', 'rating': 'reviews.rating'}, inplace=True)

# # Display the first few rows
# print("Original Data:")
# print(df.head())

# # Basic Data Cleaning
# df = df.drop_duplicates()  # Remove duplicates
# df = df.dropna(subset=['reviews'])  # Drop rows where 'reviews.text' is NaN

# # Preprocess the 'reviews.text' column
# df['reviews'] = df['reviews'].apply(preprocess_text)

# # Convert 'reviews.rating' to sentiment labels
# df['sentiment'] = df['reviews.rating'].apply(rating_to_sentiment)

# # Save the processed dataset
# df.to_csv('processed_amarev.csv', index=False)

# # Display the processed data
# print("\nProcessed Data:")
# print(df.head())
