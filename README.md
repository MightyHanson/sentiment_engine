# sentiment_engine
This project is driven by my curiosity. I love the market and I am eager to extract the emotion from the latest news and articles. Lets see what I can do.


# Introduction

The sentiment scoring engine is a comprehensive tool designed to extract, process, and analyze text from various sources to determine its sentiment. The primary objective is to assess the sentiment of articles and texts, which can be crucial for various applications, including financial analysis, market predictions, and understanding public opinion. Here we are using the model to predict the sentiment from the given article/text.

## Web Scraper
### Purpose:
Extracts text from given URLs, specifically designed for Yahoo Finance.
### Functionality:
-	Processes HTML tags, extracts the main article text, and saves it to a unified text object.
-	Uses BeautifulSoup for parsing HTML content.
-	Can extract articles based on a given time range.
-	Uses Selenium WebDriver to scroll through the Yahoo Finance page and extract more articles than the default view.
-	Can handle unexpected website structures and pop-ups.
### Expandability:
While initially designed for Yahoo Finance, the scraper can be expanded to cater to other websites in the future. The scraper is also designed to handle changes in the website structure.

## Email Filtering

### Purpose:
Filters and processes emails to extract relevant information for further analysis.
### Functionality:
-	Connects to an email account using IMAP.
-	Searches for emails based on specific criteria such as sender, date, and subject.
-	Extracts the body of the email.
-	Processes attachments, specifically PDFs, to extract their content.
-	Saves the processed emails and attachments in a structured format.
### Expandability:
While initially designed for specific email criteria, the Email Filtering system can be expanded to cater to other search criteria and attachment formats in the future.

## OCR Reader

### Purpose:
Extracts text from PDFs and processes it for further analysis.
### Functionality:
-	Converts a PDF to text. If no text is extracted directly, it uses OCR.
-	Extracts metadata such as date, author, and headline from the text.
-	Removes image captions from the main body of the text.
-	Cleans up the text extracted from PDFs, removing formatting artifacts and line breaks.
-	Extracts the main content of the article.
-	Processes all PDFs in a directory and returns a list of TextObject instances.
-	Displays the processed PDFs in a text format.
### Expandability:
While initially designed for PDFs, the OCR Reader can be expanded to cater to other document formats in the future. The reader is also designed to handle changes in the PDF structure.

## Sentiment Scoring Engine

### Purpose:
Determines the sentiment of the given articles using NLP techniques and machine learning models, including traditional methods and advanced models like BERT.
### Functionality:
-	Tokenizes, removes stop words, and lemmatizes the text.
-	Retrieves the word vector for a given word using Word2Vec.
-	Computes the document vector by averaging word vectors.
-	Assigns a sentiment label based on the count of positive and negative words.
-	Initializes and trains a Word2Vec model.
-	Extracts features and labels for the training dataset.
-	Trains and evaluates sentiment analysis models.
-	Fine-tunes BERT for sentiment analysis.
-	Uses DistilBERT for efficient sentiment analysis.

### Expandability:
The sentiment scoring engine can be expanded to include more advanced machine learning models and NLP techniques in the future. This includes the potential to integrate other transformer models or utilize ensemble methods for improved accuracy.

## Manual Labeler

### Purpose:
Allows users to manually label articles with sentiment and category.
### Functionality:
-	Loads sources of articles that have already been labeled.
-	Prompts the user to manually label an article.
-	Labels articles that haven't been labeled yet.
-	Saves the labeled articles to a CSV file.
### Expandability:
The manual labeler can be expanded to cater to other labeling tasks and can be integrated with a user interface for easier labeling in the future.

