import os
from TextObject import TextObject
import csv
class ManualLabeler:
    def __init__(self, articles=[], batch_size=3, output_directory="output/", filename="labeled_articles.csv"):
        self.articles = articles
        self.labeled_articles = []
        self.batch_size = batch_size
        self.current_index = 0
        self.output_directory = output_directory
        self.filename = filename
        self.existing_labeled_sources = self.load_existing_sources()

    def load_existing_sources(self):
        filepath = os.path.join(self.output_directory, self.filename)
        if not os.path.exists(filepath):
            return set()
        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # skip header
            return set(row[0] for row in reader)
    def label_article(self, article):
        print(article.text_content)
        category = input("Enter category (or type 'exit' to save and quit): ")
        if category.lower() == 'exit':
            return False
        sentiment = input("Enter sentiment (Positive,Negative,Netural): ")

        # Add the category and sentiment to the article's metadata dictionary
        article.metadata["category"] = category
        article.metadata["sentiment"] = sentiment

        self.labeled_articles.append(article)
        return True

    def has_more_articles(self):
        return self.current_index < len(self.articles)

    def label_unlabeled_articles(self, articles):
        for article in articles:
            # Check if the article's source is already in the labeled sources
            if article.source in self.existing_labeled_sources:
                continue
            # Check if the article already has a category and sentiment
            if 'category' in article.metadata and 'sentiment' in article.metadata:
                continue  # Skip this article if it's already labeled
            continue_labeling = self.label_article(article)
            if not continue_labeling:
                print("Saving current progress...")
                self.save_labeled_data(self.labeled_articles)
                print("Progress saved. Exiting...")
                exit(0)

    def label_articles(self, articles):
        labeled_data = []
        for article in articles:
            labeled_article = self.label_article(article)
            labeled_data.append(labeled_article)
        return labeled_data

    def save_labeled_data(self, labeled_articles):
        # Ensure the output directory exists
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        # Create the full path for the output file
        filepath = os.path.join(self.output_directory, self.filename)

        # Check if the file already exists
        if os.path.exists(filepath):
            choice = input(f"{self.filename} already exists. Do you want to (o)verwrite or (a)ppend? ").lower()
            if choice == 'o':
                mode = 'w'
            elif choice == 'a':
                mode = 'a'
            else:
                print("Invalid choice. Exiting without saving.")
                return
        else:
            mode = 'w'

        with open(filepath, mode, newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if mode == 'w':
                writer.writerow(["source", "text_content", "category", "sentiment"])
            for article in labeled_articles:
                writer.writerow(
                    [article.source, article.text_content, article.metadata['category'], article.metadata['sentiment']])

