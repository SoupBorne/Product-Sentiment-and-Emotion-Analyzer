# Product-Sentiment-and-Emotion-Analyzer
# Sentiment and Emotion Analysis Program

This repository showcases a Python program designed to perform sentiment and emotion analysis on text data, such as from files, reviews, or social media posts. It uses natural language processing (NLP) techniques and pre-trained models to provide insights into the emotional tone and sentiment expressed in the text.

## Features

- **Sentiment Analysis**: Uses VADER to classify text sentiment as positive, negative, or neutral.
- **Emotion Analysis**: Leverages the EmoRoBERTa model to identify emotions such as joy, sadness, anger, and more.
- **Top Words Identification**: Identifies words that contribute most to positive or negative sentiment.
- **Data Visualization**: Uses Matplotlib and Seaborn to visualize the analysis results.
- **Customizable Data Input**: Allows users to adapt the program for different types of text data by modifying preprocessing steps.

## How It Works

1. **Data Input**: The program reads text data from a file (`Data.txt`). The data is cleaned by converting to lowercase and removing special characters.
2. **Data Preprocessing**:
   - Uses NLTK to tokenize the text and filter out common stopwords.
   - Processes the text for further analysis.
3. **Sentiment Analysis**:
   - Uses VADER sentiment analysis to score the text as positive, negative, or neutral.
   - Identifies the top words that contribute to the overall sentiment.
4. **Emotion Analysis**:
   - Uses a pre-trained EmoRoBERTa model to detect the top five emotions in the text.
   - The results include the emotion labels and corresponding confidence scores.
5. **Visualization**:
   - Visualizes the sentiment distribution using a pie chart.
   - Displays the top detected emotions in a bar plot.
6. **Results Display**:
   - Uses Pandas to present the analysis results in a tabular format.
   - Displays the top contributing words for both positive and negative sentiment.

## Example Workflow

1. **Cleaning the Text**: The text from `Data.txt` is cleaned using `cleandata()` function to remove special characters and convert to lowercase.
2. **Sentiment and Emotion Analysis**:
   - The `Analyze_Sentiment_And_Emotion()` function performs sentiment analysis using VADER and emotion analysis using EmoRoBERTa.
   - It identifies the sentiment score and top emotions in the text.
3. **Visualizing the Results**:
   - `plot_top_emotions()` generates a bar chart of the top emotions and their scores.
   - `plot_sentiment_distribution()` creates a pie chart of sentiment distribution.
4. **Displaying the Analysis in Tabular Form**:
   - The results are displayed using the `Display_Results_With_Pandas()` function, including the sentiment score, top emotions, and contributing words.
  ### Customizing for Different Data

To adapt the program for different types of text data:

1. **Preprocess the Data**: Update the `cleandata()` function to handle any specific cleaning requirements, such as removing specific types of symbols or handling text formats.
2. **Data Source**: Change the input text file or read data from other sources, such as CSV, JSON, or online APIs.
3. **Model Settings**: Modify the settings for the EmoRoBERTa pipeline, or replace it with another model from Hugging Face's library to better suit your analysis needs.
4. **Visualization Customization**: Adjust the plotting functions to include additional data insights or different visualizations, such as word clouds or sentiment timelines.

### What this program can be used as:
The program can help businesses analyze customer feedback from sources such as Amazon reviews and social media. It does this by:
- **Breaking down feedback into sentiment scores**: Classifies feedback as positive, negative, or neutral and identifies the top emotions expressed.
- **Enabling emotion analysis**: Provides insights into the emotions connected with a product or service.
- **Detecting common issues**: Examines negative sentiment and related keywords to highlight areas needing improvement.
- **Guiding product development**: Identifies words associated with positive sentiment, helping to focus on features that customers appreciate.
- **Tracking changes over time**: Compares sentiment and emotion data over different periods to identify trends.


