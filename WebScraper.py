from TextObject import TextObject
from typing import Optional
import requests
from bs4 import BeautifulSoup
import re
import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from datetime import datetime, timedelta
# Data Source Interfaces
class WebScraper:
    def __init__(self):
        pass
    # based on Yahoo Finance for example, can extend to other source in the future
    # Yahoo Finance best other source from its structure simplicity
    # Reuters have facinating asset categories but its web structure
    def scrape(self, url: str) -> TextObject:
        # Fetch the HTML content

        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the main article content
        article_div = soup.find('div', class_='caas-body')

        # Remove all <ul> tags with the specified class
        for ul_tag in article_div.find_all('ul', class_='caas-list caas-list-bullet'):
            ul_tag.extract()

        # Remove all "Most Read from [Vendor]" parts
        most_read_pattern = re.compile(r"Most Read from \w+")
        for most_read_tag in article_div.find_all('p', string=most_read_pattern):
            most_read_tag.extract()

        article_content = article_div.get_text()

        # Extract the headline (title)
        title_div = soup.find('div', class_='caas-title-wrapper')
        title = title_div.find('h1', attrs={'data-test-locator': 'headline'}).get_text()

        # Extract the authors
        author_div = soup.find('div', class_='caas-attr-item-author')
        authors = author_div.find('span', class_='caas-author-byline-collapse').get_text()

        # Extract the date
        date_div = soup.find('div', class_='caas-attr-time-style')
        date = date_div.find('time').get_text()

        # Create and return a TextObject
        return TextObject(
            source=url,
            text_content=article_content,
            metadata={
                'Headline': title,
                'Author': authors,
                'Date': date
            }
        )
    # expand the module to achieve more power by extracting all of the articles on the main source
    # not an open source: The Wall Street Journal, Barrons
    # by default, we can only extract 10 articles
    def default_extract_article_urls(self, main_url: str) -> list:
        response = requests.get(main_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all anchor tags with the class 'js-content-viewer' that point to articles.
        article_links = soup.find_all('a', class_='js-content-viewer', href=True)
        # Extract full URLs from the anchor tags

        article_urls = set()  # Using a set to ensure uniqueness
        print(article_links)
        # in default mode, the scaper can only use 10 news
        for link in article_links:
            href = link.get('href')
            if href and (href.startswith(main_url) or href.startswith("/news/")):
                # Convert relative URLs to absolute URLs
                if href.startswith("/news/"):
                    href = main_url.rstrip('/news/') + href
                article_urls.add(href)
            # assume the main url is "https://finance.yahoo.com/",
            # which is inefficient because contaminated by other invalid source
            # if href and (href.startswith(main_url.rstrip('/')+'/news/') or href.startswith('/news/')):
            #     # Convert relative URLs to full URLs
            #     if href.startswith('/news/'):
            #         href = main_url.rstrip('/') + href
            #     article_urls.add(href)
        # print(list(article_urls))
        return list(article_urls)

    # webdriver to infinitely scroll down to catch url until the limit we set
    # solved issue: the pop up only moves once, and stop moving, can only have 10 more articles comparing with default
    # reason: If the website loads the same URLs again when you scroll down,
    # your script might think it has already collected all the URLs, even though it hasn't.
    # solution: first. Instead of scrolling to the bottom in one go, scroll the page in steps.
    # This can help in triggering the dynamic loading mechanism of the website.
    # second: Introduce Delays: After each scroll,
    # introduce a delay to give the website some time to load the new content.
    # Check for New URLs[not for yahoo]: After each scroll, check if new URLs have been added to the set.
    # If no new URLs are added after a certain number of scrolls, you can break out of the loop.
    def Webdriver_extract_article_urls(self, main_url: str, limit=50) -> list:
        # Set up the Selenium driver
        driver = webdriver.Chrome()
        driver.get(main_url)

        # Collect unique article URLs
        article_urls = set()
        last_len = 0  # To keep track of the number of URLs after each scroll
        no_new_urls_count = 0  # To count the number of scrolls with no new URLs

        while len(article_urls) < limit:
            # Scroll the page in steps, found out 1/3 each time for the scrolling can work for the automation
            for _ in range(3):
                driver.execute_script("window.scrollBy(0, window.innerHeight/3);")
                time.sleep(0.4)  # Introduce a delay

            # Wait for the page to load more articles
            WebDriverWait(driver, 15).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a.js-content-viewer"))
            )

            # Extract article URLs
            article_links = driver.find_elements(By.CSS_SELECTOR, "a.js-content-viewer")
            for link in article_links:
                href = link.get_attribute('href')
                if href and (href.startswith(main_url) or href.startswith("/news/")):
                    # Convert relative URLs to absolute URLs
                    if href.startswith("/news/"):
                        href = main_url.rstrip('/news/') + href
                    article_urls.add(href)

            # Check if new URLs have been added
            if len(article_urls) == last_len:
                no_new_urls_count += 1
            else:
                no_new_urls_count = 0

            last_len = len(article_urls)

            # If no new URLs are added after 3 scrolls, break out of the loop
            if no_new_urls_count >= 15:
                break

        driver.quit()
        return list(article_urls)[:limit]
    # v0: scrape number of articles with a limit
    # v1: upgrade to seeking a time range for the articles with a limit
    # v2: upgrade to put a time range for the articles, although the articles are listed by latest time,
    # there are lots of cases the order is not always ascending, so the current plan is to gather articles as
    # much as the limit, then filter out those within the time range. which means the num of filtered ones <= total
    # v3: yahoo finance news seems strictly follows order from latest to oldest, lets make use of that, from the
    # webdriver, we only pick the article within our defined time range
    def scrape_all_articles(self, main_url: str, limit = 50):
        article_urls = self.Webdriver_extract_article_urls(main_url,limit)
        # article_urls = self.default_extract_article_urls(main_url)
        articles = []
        for url in article_urls:
            try:
                article = self.scrape(url)  # Use self to call the scrape method
                articles.append(article)
            except Exception as e:
                print(f"Failed to scrape {url}. Reason: {e}")
        return articles
    def parse_date(self,date_str: str) -> datetime:
        return datetime.strptime(date_str, '%B %d, %Y at %I:%M %p').date()
    def scrape_all_articles_with_time_range(self, main_url: str, limit = 50, start_date=None, end_date=None):
        if not start_date:
            end_date = datetime.now().date()
            # by default I set this range to be 365 days
            start_date = end_date - timedelta(days=365)
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        article_urls = self.Webdriver_extract_article_urls(main_url,limit)
        # article_urls = self.default_extract_article_urls(main_url)
        articles = []
        for url in article_urls:
            try:
                article = self.scrape(url)  # Use self to call the scrape method
                # Parse the article's publication date
                article_date = self.parse_date(article.metadata['Date'])
                # Check if the article's date is within the desired range
                if start_date <= article_date <= end_date:
                    articles.append(article)
                # If we already have the required number of articles, break the loop
                if len(articles) == limit:
                    break
            except Exception as e:
                print(f"Failed to scrape {url}. Reason: {e}")

        return articles
    # v3 combined version of the web driver
    def scrape_articles_within_range(self, main_url: str, limit=50, start_date=None, end_date=None):
        if not start_date:
            end_date = datetime.now().date()
            # by default I set this range to be 365 days
            start_date = (end_date - timedelta(days=365))

        # Ensuring start_date and end_date are datetime.date objects
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

        # Set up the Selenium driver
        driver = webdriver.Chrome()
        driver.get(main_url)
        articles = []
        scraped_hrefs = set()  # A set to store the already scraped URLs

        last_len = 0
        no_new_urls_count = 0

        while len(articles) < limit:
            # Scroll the page in steps, found out 1/3 each time for the scrolling can work for the automation
            for _ in range(3):
                driver.execute_script("window.scrollBy(0, window.innerHeight/3);")
                time.sleep(0.6)

            # Extract article URLs
            article_links = driver.find_elements(By.CSS_SELECTOR, "a.js-content-viewer")

            for link in article_links:
                try:
                    href = link.get_attribute('href')
                    if href and (href.startswith(main_url) or href.startswith("/news/")):
                        # Convert relative URLs to absolute URLs
                        if href.startswith("/news/"):
                            href = main_url.rstrip('/news/') + href
                        # Skip the href if it's already been scraped
                        if href in scraped_hrefs:
                            continue
                        scraped_hrefs.add(href)  # Mark the href as scraped
                        # Scrape the article at the URL
                        article = self.scrape(href)


                        # Parse the article's publication date
                        article_date = self.parse_date(article.metadata['Date'])

                        # If the article's date is older than the start_date, exit
                        if article_date < start_date:
                            driver.quit()
                            return articles

                        # If the article's date is within the desired range, add it to the list
                        if start_date <= article_date <= end_date:
                            articles.append(article)

                        # If we have the required number of articles, exit
                        if len(articles) == limit:
                            driver.quit()
                            return articles

                except Exception as e:
                    print(f"Failed to scrape {href}. Reason: {e}")

            # Check if new URLs have been added
            if len(articles) == last_len:
                no_new_urls_count += 1
            else:
                no_new_urls_count = 0

            last_len = len(articles)

            # If no new articles are added after 3 scrolls, break out of the loop
            if no_new_urls_count >= 15:
                break

        driver.quit()
        return articles

    def display_articles(self, articles: list):
        for article in articles:
            print("Source:", article.source)
            print("Text Content:", article.text_content)
            print("Metadata:", article.metadata)
            print("--------------------------------------------------")

    def display_articles_txt(self, articles: list, output_directory='output',
                             output_filename= 'articles.txt',overwrite: bool = False):
        # Ensure the output directory exists
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Read the existing articles.txt and extract the URLs if not overwriting
        existing_urls = set()
        if not overwrite:
            try:
                with open(os.path.join(output_directory, output_filename), 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    for line in lines:
                        if line.startswith("Source: "):
                            existing_urls.add(line.split("Source: ")[1].strip())
            except FileNotFoundError:
                # If the file doesn't exist, we'll create it later
                pass

        # Determine the mode based on the overwrite flag
        mode = 'w' if overwrite else 'a'

        # Write to articles.txt
        with open(os.path.join(output_directory, output_filename), mode, encoding='utf-8') as file:
            for article in articles:
                if overwrite or article.source not in existing_urls:
                    file.write("Source: " + article.source + "\n")
                    file.write("Text Content: " + article.text_content + "\n")
                    file.write("Metadata: " + str(article.metadata) + "\n")
                    file.write("--------------------------------------------------\n")
