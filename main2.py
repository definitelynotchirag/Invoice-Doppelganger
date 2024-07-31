import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re,os
import numpy as np
from PIL import Image
import cv2
import pdf2image
import imagehash
import streamlit as st

from io import StringIO
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
from lxml import html
import bs4


class InvoiceComparer:
    def __init__(self):
        self.database = []
        self.vectorizer = TfidfVectorizer()

    def extract_text(self, pdf_path):
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text

    def extract_metadata(self, pdf_path):
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            metadata = reader.metadata
        return metadata

    def extract_table_style(self, pdf_path):
        output = StringIO()

        with open(pdf_path, 'rb') as pdf_file:
            extract_text_to_fp(pdf_file, output, laparams=LAParams(), output_type='html', codec=None)
        with open('./example.html','w') as html_file:
            html_file.write(output.getvalue())

        with open('./example.html') as html_file:
            soup = bs4.BeautifulSoup(html_file,"html.parser")

        divs = soup.find_all('div')


        data = []
        for div in divs:
            style = div.get('style', '')
            if 'writing-mode:lr-tb' in style:
                text = div.get_text(strip=True)
                data.append(style)

        return data

    def preprocess_text(self, text):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        return text

    def extract_features(self, text, pdf_path):
        invoice_number = re.search(r'GLN (\d+)', text, re.IGNORECASE)
        date = re.search(r'\d{2}\.\d{2}\.\d{4}', text, re.IGNORECASE)
        styles = self.extract_table_style(pdf_path)
        metadata = self.extract_metadata(pdf_path)
        pdf_files = re.findall(r'\b\w+\.pdf\b', pdf_path, re.IGNORECASE)

        features = ""
        if invoice_number:
            features += f"Invoice number: {invoice_number.group(1)} "
        if date:
            features += f"Date: {date.group(1)} "
        if styles:
            features += f"Styles: {styles}"
        if metadata:
            features += f"Metadata: {metadata}"
        if pdf_files:
            features += f"PDF files: {pdf_files}"

        return features + text


    def pdf_to_image(self, pdf_path):
        images = pdf2image.convert_from_path(pdf_path)
        return images[0]  # We'll just use the first page for simplicity

    def calculate_image_similarity(self, img1, img2):
        hash1 = imagehash.phash(img1)
        hash2 = imagehash.phash(img2)
        return 1 - (hash1 - hash2) / len(hash1.hash) ** 2

    def add_to_database(self, pdf_path):
        text = self.extract_text(pdf_path)
        processed_text = self.preprocess_text(text)
        features = self.extract_features(processed_text, pdf_path)
        image = self.pdf_to_image(pdf_path)
        self.database.append((pdf_path, features, image))

    def find_most_similar(self, input_pdf):
        input_text = self.extract_text(input_pdf)
        input_processed = self.preprocess_text(input_text)
        input_features = self.extract_features(input_processed, input_pdf)
        input_image = self.pdf_to_image(input_pdf)

        all_texts = [input_features] + [features for _, features, _ in self.database]
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)

        text_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        image_similarities = []
        for _, _, db_image in self.database:
            image_similarities.append(self.calculate_image_similarity(input_image, db_image))

        combined_similarities = 0.7 * text_similarities + 0.3 * np.array(image_similarities)

        most_similar_index = np.argmax(combined_similarities)

        return self.database[most_similar_index][0], combined_similarities[most_similar_index]

if __name__ == '__main__':
    comparer = InvoiceComparer()

    for pdfs in os.listdir("./training_data/"):
        comparer.add_to_database(f"./training_data/{pdfs}")

    tfiles = []
    for pdfs in os.listdir("./testing_data"):
        tfiles.append(pdfs)
        most_similar, similarity_score = comparer.find_most_similar(f"./testing_data/{pdfs}")
        os.remove('./example.html')
        print(f"Input invoice: {pdfs}")
        print(f"Most similar invoice: {most_similar}")
        print(f"Similarity score: {similarity_score}")
