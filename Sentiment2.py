
"""
#@author: Corey Douglas
"""
import string #To clean text by removing special characters
from nltk.tokenize import word_tokenize #To tokenize Text into individual words
from nltk.corpus import stopwords #dataset of stopwords(Removes words without sentiment)
from nltk.sentiment.vader import SentimentIntensityAnalyzer #For sentiment Analysis
import pandas as pd #To create dataframes
from transformers import pipeline #To access pretrained model for emotion analysis
import tensorflow #The emoroberta pretrained model depends on tensorflow
import matplotlib.pyplot as plt #To visualize data
import seaborn as sns #Enahnces matplotlib visualizations
from collections import Counter #To count word frequencies and identify top words


#The clean data function will remove special charcters and convert to lowercase.
def cleandata(data):
    # Only clean strings; check for string type before processing
    if isinstance(data, str):
        lower_data = data.lower() #convert text to lowercase
        cleaned_data = lower_data.translate(str.maketrans('', '', string.punctuation))
        return cleaned_data
    return data  # Return non-string data as is
#Identify words that contribute to the sentiment of the text
def find_contributing_words(sentiment_text):
    #Start VADER Sentiment Analyzer
    sia = SentimentIntensityAnalyzer()
    words = word_tokenize(sentiment_text) #Tokenize text into words
    stop_words = set(stopwords.words('english')) #Get stopwords in english
    filtered_words = [word for word in words if word.lower() not in stop_words] #filer stopwords out
    # Analyze sentiment for each word
    contributing_words = {'positive': [], 'negative': []} #stores contributing words as either positive or negative
    for word in filtered_words:
        word_score = sia.polarity_scores(word) #get the sentiment score of a word
        if word_score['compound'] > 0.05:  # Positive
            contributing_words['positive'].append(word)
        elif word_score['compound'] < -0.05:  # Negative
            contributing_words['negative'].append(word)
            
    # Get the top 5 positive and negative words
    top_positive = Counter(contributing_words['positive']).most_common(5)
    top_negative = Counter(contributing_words['negative']).most_common(5)
    
    #Return a dictionary with the top positive or negative
    return {
        'positive': [word for word, _ in top_positive],
        'negative': [word for word, _ in top_negative] }
    return contributing_words
#Read in the text file and clean it 
text = open("Data.txt", encoding="utf-8").read()

cleaned_text = cleandata(text)


def Analyze_Sentiment_And_Emotion(sentiment_text):
    # Initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Get sentiment score
    score = sia.polarity_scores(sentiment_text)
    neg = cleandata(score['neg'])
    pos = cleandata(score['pos'])
    
    # Determine overall sentiment
    if neg > pos:
        sentiment = "Negative sentiment"
    elif pos > neg:
        sentiment = "Positive sentiment"
    else:
        sentiment = "Neutral sentiment"
        
    
    # Initialize emotion analysis model(EmoRoberta)
    emotion_pipeline = pipeline('sentiment-analysis', model="arpanghoshal/EmoRoBERTa", top_k=None, truncation=True)
    


    # Get emotion analysis
    emotion = emotion_pipeline(sentiment_text)

   # Sort emotions by score and take the top 5
    sorted_emotions = sorted(emotion[0], key=lambda x: x['score'], reverse=True)[:5]
    
    # Extract emotion labels and scores
    top_emotions = [(cleandata(e['label']), round(e['score'], 4)) for e in sorted_emotions]
    
    # Find contributing words
    contributing_words = find_contributing_words(sentiment_text)
    
    # Return the results as a dictionary
    return {
        'sentiment': sentiment,
        'sentiment_score': score,
        'top_emotions': top_emotions,
        'contributing_words': contributing_words
    }
def Display_Results_With_Pandas(analysis_results):
    # Create a dictionary to store the data for the DataFrame
    data = {
        'Sentiment': [analysis_results['sentiment']],
        'Sentiment Score': [analysis_results['sentiment_score']]
    }
    
    # Add the top emotions and their scores to the data
    for i, (emotion, score) in enumerate(analysis_results['top_emotions']):
        data[f'Emotion {i+1}'] = [emotion]
        data[f'Emotion {i+1} Score'] = [score]
    
    # Convert the dictionary into a pandas DataFrame
    df = pd.DataFrame(data)
    
    pd.set_option('display.max_columns', None)
    
    # Format the sentiment scores
    df['Sentiment Score'] = df['Sentiment Score'].apply(lambda x: {k: f"{v:.4f}" for k, v in x.items()})

    # Format the emotion scores
    for i in range(1, 6):
        df[f'Emotion {i} Score'] = df[f'Emotion {i} Score'].astype(float).map('{:.4f}'.format)
    
    df.columns = [
        'Sentiment',
        'Sentiment Score',
        '1st Emotion', '1st Emotion Score',
        '2nd Emotion', '2nd Emotion Score',
        '3rd Emotion', '3rd Emotion Score',
        '4th Emotion', '4th Emotion Score',
        '5th Emotion', '5th Emotion Score'
    ]

    # Display the DataFrame
    print(df)
    
    # Display contributing words
    contributing_words = analysis_results['contributing_words']
    print("\nContributing Words:")
    print(f"Positive Words: {', '.join(contributing_words['positive']) if contributing_words['positive'] else 'None'}")
    print(f"Negative Words: {', '.join(contributing_words['negative']) if contributing_words['negative'] else 'None'}")
    
    return df
def plot_top_emotions(analysis_results):
    # Extract top emotions and their scores
    emotions = [emotion[0] for emotion in analysis_results['top_emotions']]
    scores = [emotion[1] for emotion in analysis_results['top_emotions']]

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=scores, y=emotions, palette='viridis')

    # Add title and labels
    plt.title('Top Emotions and Their Scores')
    plt.xlabel('Score')
    plt.ylabel('Emotions')

    # Show the plot
    plt.show()
def plot_sentiment_distribution(analysis_results):
    # Define sentiment counts
    sentiment_counts = {
        'Positive': analysis_results['sentiment_score']['pos'],
        'Negative': analysis_results['sentiment_score']['neg'],
        'Neutral': analysis_results['sentiment_score']['neu']
    }

    # Create a pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%', startangle=140)
    
    # Add title
    plt.title('Sentiment Distribution')
    
    # Show the plot
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
    plt.show()
#Analyze the cleaned text for sentiment and emotion
results = Analyze_Sentiment_And_Emotion(cleaned_text)
#Plot the top emotions and the sentiment
plot_top_emotions(results)
plot_sentiment_distribution(results)
#Display the results in dataframe formatting
Display_Results_With_Pandas(results)