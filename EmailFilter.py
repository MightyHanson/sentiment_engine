# email filter package
from TextObject import TextObject
import imaplib
from email import policy
from email.parser import BytesParser
from datetime import datetime, timedelta
import email.utils
from bs4 import BeautifulSoup
import re
import os
import time

class EmailFilter:
    def __init__(self, email_address: str, password: str, imap_server: str = "imap-mail.outlook.com"
                 , imap_port: int = 993,output_directory: str = 'output/',output_filename: str = 'emails.txt',
                 news_sources: list = ["(Reuters)", "(Bloomberg)"]):
        self.email_address = email_address
        self.password = password
        self.imap_server = imap_server
        self.imap_port = imap_port
        self.output_directory = output_directory
        self.output_filename = output_filename
        self.news_sources = news_sources

    # Add a method to extract the author:
    def extract_authors(self, text):
        # Split the text by commas, 'and', or semicolons
        parts = re.split(',| and |;', text)

        authors = []
        for part in parts:
            # Remove any locations and unwanted text
            author = re.sub(r"\bin\b [\w\s]+|Reporting by|Writing by|Editing by|Additional reporting by", "",
                            part).strip()
            # Remove unwanted characters like "<"
            author = author.replace("<", "").strip()
            # If the author's name has at least two parts, add it to the list
            if len(author.split()) >= 2:
                authors.append(author)

        # Remove duplicates and return
        return list(set(authors))

    def remove_and_extract_authors(self, body: str) -> tuple[str, list[str]]:
        authors = []
        patterns = ["Reporting by", "Writing by", "Editing by"]
        for pattern in patterns:
            if pattern in body:
                # Extract the section starting from the pattern
                section = body.split(pattern, 1)[1]
                # Extract authors from the section
                authors += self.extract_authors(section)
                # Remove the section from the body
                body = body.rsplit(pattern, 1)[0]
        return body.strip(), authors

    def format_email_body(self, email_body: str, unwanted_start_phrases: list) -> str:
        # Remove URLs from the article
        email_body = re.sub(r'http\S+', '', email_body)

        # Convert HTML to plain text (if any HTML tags are present)
        soup = BeautifulSoup(email_body, 'html.parser')
        email_body = soup.get_text()

        # Remove lines that start with unwanted phrases
        lines = email_body.split('\n')
        lines = [line for line in lines if not any(line.startswith(phrase) for phrase in unwanted_start_phrases)]
        email_body = '\n'.join(lines).strip()
        # Replace newline characters with spaces
        email_body = email_body.replace('\n', ' ')
        # Remove any extra spaces
        email_body = ' '.join(email_body.split())

        return email_body
    # v0 extract all of the emails from the account
    # v1 adding fiter, from a safe source list, within a time range, and with a limit of the emails being retrieved
    # by default, I set the code to retrieve the email from the past 7 days
    # v2 remove the unwanted url, and some unwanted words from the email body; and unify the format with the previous one
    def extract_emails(self, sources_limits: dict, start_date: datetime = None,
                       end_date: datetime = None, unwanted_start_phrases: list = None) -> list[TextObject]:
        # If start_date and end_date are not provided, default to the past 7 days
        # issue: can't compare offset-naive and offset-aware datetimes -- solution: they should be the same
        if not start_date and not end_date:
            # make it includes the whole day
            end_date = datetime.now().replace(hour=23, minute=59, second=59, microsecond=999999, tzinfo=None)
            start_date = end_date - timedelta(days=7)
        # Adjust start_date and end_date to include the entire day
        if start_date:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        if end_date:
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)

        # Connect to the email server
        mail = imaplib.IMAP4_SSL('imap-mail.outlook.com')
        mail.login(self.email_address, self.password)
        mail.select('inbox')

        # Fetch all email IDs
        result, email_ids = mail.search(None, "ALL")
        email_ids = email_ids[0].split()

        text_objects = []

        for email_id in email_ids:
            result, email_data = mail.fetch(email_id, '(RFC822)')
            raw_email = email_data[0][1]
            email_message = BytesParser(policy=policy.default).parsebytes(raw_email)

            # Extract email details

            from_address = email.utils.parseaddr(email_message['From'])[1]
            email_date = email.utils.parsedate_to_datetime(email_message['Date']).replace(tzinfo=None)

            # Check if the email is from a desired source and within the date range
            if from_address in sources_limits.keys() and start_date <= email_date <= end_date:
                # Extract the email body
                for part in email_message.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode(errors='replace')

                        # Find the first occurrence of any news source in the body
                        # start from beginning if no source is found -- if sources are determined, it can speed up
                        body_start_index = min(
                            [body.find(source) for source in self.news_sources if body.find(source) != -1], default=-1)
                        if body_start_index != -1:
                            body = body[body_start_index:]
                        body = self.format_email_body(body, unwanted_start_phrases)
                        body, authors = self.remove_and_extract_authors(body)
                        # extract the headline and author
                        metadata = {'Date': str(email_date), 'Headline': email_message['Subject'], 'Author': authors}
                        text_objects.append(TextObject(source=from_address, text_content=body, metadata=metadata))
                        sources_limits[from_address] -= 1  # Decrement the limit for the source

                        # If the limit for the source is reached, remove it from the sources_limits dictionary
                        if sources_limits[from_address] == 0:
                            del sources_limits[from_address]

        mail.logout()

        return text_objects
    # format the email body to make it look like the web scraper did


    def display_emails(self, emails: list):
        for email in emails:
            print("Source:", email.source)
            print("Text Content:", email.text_content)
            print("Metadata:", email.metadata)
            print("--------------------------------------------------")

    def display_emails_txt(self, emails: list):
        # Ensure the output directory exists
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        # default on windows is gbk encoding
        with open(os.path.join(self.output_directory, self.output_filename), 'w', encoding='utf-8') as file:
            for email in emails:
                file.write("Source: " + email.source + "\n")
                file.write("Text Content: " + email.text_content + "\n")
                file.write("Metadata: " + str(email.metadata) + "\n")
                file.write("--------------------------------------------------\n")
