from WebScraper import WebScraper
from EmailFilter import EmailFilter
from OCRReader import OCRReader
from SentimentScoringEngine import SentimentScoringEngine
from ManualLabeler import ManualLabeler
from helper_func import *

class MainProcessor:
    def __init__(self):
        self.scraper = WebScraper()
        self.email_filter = EmailFilter("hanwangtest123@hotmail.com", "hanwang123")
        self.ocr_reader = None
        self.engine = None
        self.labeler = ManualLabeler()

    def scrape_articles(self, main_url, limit, start_date, end_date,output_directory=None,output_filename=None):
        # v0 basic scrape a url
        # self.scraper.scrape("https://finance.yahoo.com/news/allstate-pay-90-million-shareholder-150835962.html.")
        # v1 scrape latest limit number of urls
        # all_articles = self.scraper.scrape_all_articles(main_url, limit)
        # v2 scrape, then fileter the article within time range
        # all_articles = self.scraper.scrape_all_articles_with_time_range(main_url, limit, start_date,end_date)
        # v3 scrape while keep within the time range [not recommend when network is slow]
        all_articles = self.scraper.scrape_articles_within_range(main_url, limit, start_date, end_date)
        self.scraper.display_articles_txt(all_articles, output_directory,output_filename,overwrite=False)
        self.scraper.display_articles(all_articles)

    def process_emails(self, account, password,imap_server, imap_port,sources_limits, start_date,
                       end_date, unwanted_phrases=None,
                       output_directory: str = 'output/', output_filename: str = 'emails.txt',
                       news_sources: list = []):
        emails = EmailFilter(account, password,imap_server, imap_port,
                             output_directory,output_filename,news_sources)
        emails = self.email_filter.extract_emails(sources_limits, start_date, end_date, unwanted_phrases)
        self.email_filter.display_emails(emails)
        self.email_filter.display_emails_txt(emails)

    def process_pdfs(self, ocr_directory, ocr_output_directory='output/', ocr_output_filename='pdfs.txt'):
        self.ocr_reader = OCRReader(ocr_directory, ocr_output_directory, ocr_output_filename)
        pdf_objects = self.ocr_reader.process_pdfs()
        self.ocr_reader.display_pdfs_txt(pdf_objects)

    def analyze_sentiments(self, input_comb_text, model_path=None):
        self.engine = SentimentScoringEngine(model_path)
        # For Word2Vec
        print("Results using Word2Vec embeddings:")
        logistic_model_w2v, random_forest_model_w2v = self.engine.train_and_evaluate(input_comb_text, use_bert=False)
        print("Word2Vec - logistic_model: {}, random_forest_model: {}".format(logistic_model_w2v,
                                                                              random_forest_model_w2v))
        # For BERT slow but useful, DistilBERT can speed up
        print("\nResults using BERT embeddings:")
        logistic_model_bert, random_forest_model_bert = self.engine.train_and_evaluate(input_comb_text, use_bert=True)
        print("BERT - logistic_model: {}, random_forest_model: {}".format(logistic_model_bert, random_forest_model_bert))

    def label_unlabeled_articles(self, articles,output_directory,filename):
        self.labeler = ManualLabeler(output_directory=output_directory, filename="labeled_articles.csv")
        return self.labeler.label_unlabeled_articles(articles)

    def run(self):
        # both work
        # main_url = "https://finance.yahoo.com/"
        main_url = "https://finance.yahoo.com/news/"
        start_date = '2024-05-20'
        end_date = '2024-05-28'
        account = "hanwangtest123@hotmail.com"
        password = "hanwang123"
        imap_server = "imap-mail.outlook.com"
        imap_port = 993
        sources_limits = {
            "hanwang616@hotmail.com": 10,
            "hansonwang616@gmail.com": 15
        }
        unwanted_phrases = ["Our Standards:", "Another unwanted phrase:", "..."]
        output_directory = 'output/'
        w2v_model_dir = 'w2v_model/'
        news_sources = ["(Reuters)", "(Bloomberg)"]

        self.scrape_articles(main_url, 500, start_date, end_date,output_directory,'articles.txt')
        # self.process_emails(account, password,imap_server, imap_port,
        #                     sources_limits, start_date, end_date, unwanted_phrases,
        #                     output_directory,'emails.txt',news_sources)
        self.process_pdfs("F:\pythonProject1\pdf_collection", output_directory, 'pdfs.txt')
        concatenate_txt_files(output_directory, 'combine.txt')

        input_comb_text = extract_text_content(os.path.join(output_directory, 'combine.txt'))
        self.analyze_sentiments(input_comb_text,os.path.join(w2v_model_dir,'my_word2vec_model.model'))

        # future development with labeled
        # comb_text_obj = read_text_objects_from_txt(os.path.join(output_directory, 'combine.txt'))
        # labeled_articles = self.label_unlabeled_articles(comb_text_obj,output_directory=output_directory,filename="labeled_articles.csv")
        # self.labeler.save_labeled_data(labeled_articles)


if __name__ == "__main__":
    processor = MainProcessor()
    processor.run()
