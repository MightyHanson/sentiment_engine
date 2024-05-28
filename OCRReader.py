from TextObject import TextObject
# pdf
import PyPDF2
from pdf2image import convert_from_path
import pytesseract
import re
import os
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

class OCRReader:
    def __init__(self, directory: str, output_directory: str = 'output/', output_filename: str = 'pdfs.txt'):
        self.directory = directory
        self.output_directory = output_directory
        self.output_filename = output_filename
    def pdf_to_text(self, pdf_path: str) -> str:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
            # If no text was extracted, use OCR
            if not text.strip():
                try:
                    from PIL import Image
                except ImportError:
                    raise ImportError("Please install PIL/Pillow to use OCR on images.")
                text = pytesseract.image_to_string(Image.open(pdf_path))
            return text

    def extract_metadata(self, text: str) -> dict:
        def preprocess_date_str(date_str: str) -> str:
            # Preprocess the date_str to coordinate with the time zone examples
            time_zones = ['ET', 'EST', 'CT', 'PT', 'MT', 'EDT']
            for tz in time_zones:
                spaced_tz = ' '.join(tz)
                date_str = date_str.replace(spaced_tz, tz)
            return date_str
        # Extract date
        # issue solved [APM] can only deal with those in cap, we also need the lowers
        # issue solved, space between the ET
        # expand to other time zone cases
        date_formats = [
            ('%B %d, %Y at %I:%M %p EDT', r'(\w+ \d{1,2}, \d{4} at \d{1,2}:\d{2} [APMapm]{2} E\s*D\s*T)'),
            ('%b. %d, %Y %I:%M %p ET', r'(\w{3}\. \d{1,2}, \d{4} \d{1,2}:\d{2} [APMapm]{2} E\s*T)'),
            ('%b. %d, %Y %I:%M %p EST', r'(\w{3}\. \d{1,2}, \d{4} \d{1,2}:\d{2} [APMapm]{2} E\s*S\s*T)'),
            ('%b. %d, %Y %I:%M %p CT', r'(\w{3}\. \d{1,2}, \d{4} \d{1,2}:\d{2} [APMapm]{2} C\s*T)'),
            ('%b. %d, %Y %I:%M %p PT', r'(\w{3}\. \d{1,2}, \d{4} \d{1,2}:\d{2} [APMapm]{2} P\s*T)'),
            ('%b. %d, %Y %I:%M %p MT', r'(\w{3}\. \d{1,2}, \d{4} \d{1,2}:\d{2} [APMapm]{2} M\s*T)')
        ]
        date = None
        date_str = None
        date_str_ori = None
        for date_format, pattern in date_formats:
            date_match = re.search(pattern, text)
            if date_match:
                date_str_ori = date_match.group(1)
                # expand to other time zone
                # date_str = date_str_ori.replace('E T', 'ET').replace('E  T', 'ET')  # Preprocess the date_str
                date_str = preprocess_date_str(date_str_ori)
                date_obj = datetime.strptime(date_str, date_format)
                date = date_obj.strftime('%Y-%m-%d %H:%M:%S')
                break

        # Extract author (line above the date that contains at least two words)
        # The reason using the original data_str is that if we use the revised version as the date format did
        # we would lose track of its location in the original text
        date_line_index = text.find(date_str_ori)
        lines_before_date = text[:date_line_index].strip().split('\n')

        author_line = next((line for line in reversed(lines_before_date) if len(line.split()) >= 2), None)
        if author_line:
            # Remove "By" or "by" prefix and split by "and"
            authors = [author.strip() for author in author_line.replace("By", "").replace("by", "").split("and")]
        else:
            authors = []

        # Extract headline (all lines before the author)
        headline = text[:text.find(author_line)].strip().replace('\n', ' ')
        headline = re.sub(r'\s+', ' ', headline)  # Replace multiple spaces with a single space

        return {
            'Headline': headline,
            'Author': authors,
            'Date': date,
            'date_str_ori':date_str_ori
        }

    def remove_image_captions(self,main_body: str) -> str:
        # Remove lines that are likely image captions based on their length and potential keywords
        # The pattern captures lines that contain keywords like "PHOTO", "Source", "Note", etc.
        # and are of a reasonable length to be considered as captions.
        # Patterns for removing captions around the image
        image_captions_patterns = [
            r'".*?"\s*$',  # Captions in quotes
            r'^.*?PHOTO:.*?$', # Remove the lines contain with "PHOTO:"
            r'^.*?IMAGES.*?$',  # Remove the lines contain with "IMAGES", specifically found from one article
            r'^\s*Source:.*$',  # Remove lines starting with "Source:"
            r'^\s*Sources:.*$',  # Remove lines starting with "Sources:"
            r'^\s*Note:.*$',  # Remove lines starting with "Note:"
            r'^\s*PHOTO:.*$',  # Remove lines starting with "PHOTO:"
        ]

        for pattern in image_captions_patterns:
            main_body = re.sub(pattern, '', main_body, flags=re.IGNORECASE | re.MULTILINE)

        # Remove any double newlines that might be left after removing captions
        main_body = re.sub(r'\n{2,}', '\n', main_body).strip()

        return main_body
    # The main challenge with PDFs is that they might contain formatting artifacts,
    # line breaks, and other inconsistencies.
    def clean_pdf_text(self,text):
        # Convert to plain text if containing HTML
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()

        # Remove unwanted line breaks (by breaking each line to clean up) and concatenate lines
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            if line:
                cleaned_lines.append(line.strip())

        # Remove headers, footers, and page numbers
        # Assuming headers/footers are repeating lines, we can count occurrences and remove lines that appear too often
        line_counts = {line: cleaned_lines.count(line) for line in set(cleaned_lines)}
        cleaned_lines = [line for line in cleaned_lines if line_counts[line] <= 2 and not line.isdigit()]

        # Join all lines into a single paragraph
        single_paragraph = ' '.join(cleaned_lines)

        return single_paragraph
    def extract_main_body(self, text: str, date_str = "") -> str:
        # Extract main body (all lines after the date)
        main_body_start = text.find(date_str) + len(date_str)
        main_body = text[main_body_start:].strip()
        # Remove patterns like "· 2 min read" from the main body
        main_body = re.sub(r'·? \d+ min read', '', main_body)
        # Remove patterns like "(Reporting by [Name]; Editing by [Name])" from the main body
        # main_body = re.sub(r'\(Reporting by .*?; Editing by .*?\)', '', main_body)
        main_body = self.remove_image_captions(main_body)
        # clean pdf text,
        main_body = self.clean_pdf_text(main_body)


        return main_body.strip()

    def process_pdfs(self) -> list[TextObject]:
        text_objects = []
        for pdf_file in os.listdir(self.directory):
            if pdf_file.endswith('.pdf'):
                pdf_path = os.path.join(self.directory, pdf_file)
                text = self.pdf_to_text(pdf_path)

                # Extract metadata
                # adding one case that, if we fail to extract the information,
                # we put everything into the main body
                try:
                    metadata = self.extract_metadata(text)
                    main_body = self.extract_main_body(text, metadata['date_str_ori'])
                    del metadata['date_str_ori']
                except:
                    metadata = {'Headline': '',
                                'Author': [],
                                'Date': ''}
                    main_body = self.extract_main_body(text)


                # Create a TextObject instance
                text_object = TextObject(source=pdf_file, text_content=main_body, metadata=metadata)
                text_objects.append(text_object)

        return text_objects

    def display_pdfs_txt(self, pdfs: list):
        # Ensure the output directory exists
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        # default on windows is utf-8 encoding
        with open(os.path.join(self.output_directory, self.output_filename), 'w', encoding='utf-8') as file:
            for pdf in pdfs:
                file.write("Source: " + pdf.source + "\n")
                file.write("Text Content: " + pdf.text_content + "\n")
                file.write("Metadata: " + str(pdf.metadata) + "\n")
                file.write("--------------------------------------------------\n")