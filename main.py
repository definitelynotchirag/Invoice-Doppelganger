import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re, os
import numpy as np
from PIL import Image
import pdf2image
import imagehash
import pickle

from io import StringIO
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
from lxml import html
import bs4

class InvoiceComparer:
    def __init__(self, vectorizer_path, database_path, training_data_path):
        self.vectorizer_path = vectorizer_path
        self.database_path = database_path
        self.training_data_path = training_data_path
        self.vectorizer = None
        self.database = []
        self.load_or_train_model()

    def load_or_train_model(self):
        if os.path.exists(self.vectorizer_path) and os.path.exists(self.database_path):
            self.load_model()
            self.check_for_new_files()
        else:
            self.train_model()

    def load_model(self):
        # Load the trained vectorizer and database
        with open(self.vectorizer_path, 'rb') as model_file:
            self.vectorizer = pickle.load(model_file)
        with open(self.database_path, 'rb') as db_file:
            self.database = pickle.load(db_file)

    def train_model(self):
        # Train the model on the fly using the training data
        self.database = []
        for pdfs in os.listdir(self.training_data_path):
            self.add_to_database(os.path.join(self.training_data_path, pdfs))

        all_texts = [features for _, features, _ in self.database]
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(all_texts)

        # Save the trained vectorizer and database
        with open(self.vectorizer_path, 'wb') as model_file:
            pickle.dump(self.vectorizer, model_file)
        with open(self.database_path,'wb') as db_file:
            pickle.dump(self.database, db_file)

    def check_for_new_files(self):
        # Check for new files in the training data directory
        db_files = []
        existing_files = {entry[0] for entry in self.database}
        for file in existing_files:
             path = os.path.basename(file)
             db_files.append(path)

        new_files = set(os.listdir(self.training_data_path)) - set(db_files)

    
        if new_files:
            print("New Training Data detected. Retraining the model...")
            self.train_model()

    def extract_text_and_metadata(self, pdf_path):
        # Extract text and metadata from PDF using PyPDF2
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            metadata = reader.metadata
            
        pdf_files = re.findall(r'\b\w+\.pdf\b', pdf_path, re.IGNORECASE)
        
        return text, metadata, pdf_files[0]

    def extract_table_style(self, pdf_path):
        # Extract table styles from PDF using pdfminer and BeautifulSoup
        output = StringIO()

        with open(pdf_path, 'rb') as pdf_file:
            extract_text_to_fp(pdf_file, output, laparams=LAParams(), output_type='html', codec=None)

        soup = bs4.BeautifulSoup(output.getvalue(), "html.parser")
        divs = soup.find_all('div')

        data = []
        for div in divs:
            style = div.get('style', '')
            if 'writing-mode:lr-tb' in style:
                text = div.get_text(strip=True)
                data.append(style)

        return data

    def preprocess_text(self, text):
        # Preprocess text by removing non-alphanumeric characters and converting to lowercase
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        return text

    def extract_features(self, text, metadata, styles, pdf_file):
        # Extract features from text and PDF metadata
        invoice_number = re.search(r'GLN (\d+)', text, re.IGNORECASE)
        date = re.search(r'\d{2}\.\d{2}\.\d{4}', text, re.IGNORECASE)
        

        features = ""
        if invoice_number:
            features += f"Invoice number: {invoice_number.group(1)} "
        if date:
            features += f"Date: {date.group(1)} "
        if styles:
            features += f"Styles: {styles}"
        if metadata:
            features += f"Metadata: {metadata}"

        if pdf_file:
            features += f"Name: {pdf_file}"
        return features + text

    def pdf_to_image(self, pdf_path):
        # Convert the first page of the PDF to an image using pdf2image
        images = pdf2image.convert_from_path(pdf_path)
        return images[0]  # We'll just use the first page for simplicity

    def calculate_image_similarity(self, img1, img2):
        # Calculate image similarity using perceptual hashing
        hash1 = imagehash.phash(img1)
        hash2 = imagehash.phash(img2)
        return 1 - (hash1 - hash2) / len(hash1.hash) ** 2

    def add_to_database(self, pdf_path):
        # Add invoice data to the database
        text, metadata, pdf_file = self.extract_text_and_metadata(pdf_path)
        processed_text = self.preprocess_text(text)
        styles = self.extract_table_style(pdf_path)
        features = self.extract_features(processed_text, metadata, styles, pdf_file)
        image = self.pdf_to_image(pdf_path)
        self.database.append((pdf_path, features, image))

    def find_most_similar(self, input_pdf):
        # Find the most similar invoice in the database
        input_text, input_metadata, input_pdf_file = self.extract_text_and_metadata(input_pdf)
        input_processed = self.preprocess_text(input_text)
        input_styles = self.extract_table_style(input_pdf)
        input_features = self.extract_features(input_processed, input_metadata, input_styles, input_pdf_file)
        input_image = self.pdf_to_image(input_pdf)

        all_texts = [input_features] + [features for _, features, _ in self.database]
        tfidf_matrix = self.vectorizer.transform(all_texts)

        text_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        image_similarities = []
        for _, _, db_image in self.database:
            image_similarities.append(self.calculate_image_similarity(input_image, db_image))

        # Combine text and image similarities (you can adjust the weights)
        combined_similarities = 0.7 * text_similarities + 0.3 * np.array(image_similarities)

        most_similar_index = np.argmax(combined_similarities)

        return self.database[most_similar_index][0], combined_similarities[most_similar_index]

if __name__ == '__main__':
    comparer = InvoiceComparer('./models/vectorizer.pkl', './models/database.pkl', './training_data')

    tfiles = []
    # Compare each test PDF with the database and print the most similar invoice
    for pdfs in os.listdir("./testing_data"):
        tfiles.append(pdfs)
        most_similar, similarity_score = comparer.find_most_similar(f"./testing_data/{pdfs}")
        print(f"Input invoice: {pdfs}")
        print(f"Most similar invoice: {most_similar}")
        print(f"Similarity score: {similarity_score}")
