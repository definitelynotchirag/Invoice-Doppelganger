import streamlit as st
import os, re
from main import InvoiceComparer
import time

comparer = InvoiceComparer('./models/vectorizer.pkl', './models/database.pkl', './training_data')

st.title("Invoice Doppelganger")

tfiles = []
for pdfs in os.listdir("./testing_data"):
    tfiles.append(pdfs)

option = st.selectbox(label="Select A Testing File", index=None, placeholder="Select An Invoice", options=tfiles)

for pdfs in os.listdir("./training_data/"):
    comparer.add_to_database(f"./training_data/{pdfs}")


def runit():
    with st.spinner(text="In progress"):
        bar = st.progress(25)
        most_similar, similarity_score = comparer.find_most_similar(f"./testing_data/{option}")
        print(f"Input invoice: {option}")
        print(f"Most similar invoice: {most_similar}")
        bar.progress(100)
        print(f"Similarity score: {similarity_score}")
        pdf_files = re.findall(r'\b\w+\.pdf\b', most_similar, re.IGNORECASE)
        st.success(f"### Most similar invoice: {pdf_files[0]}\n")
        st.subheader(f"Similarity score: {similarity_score}\n\n")

if option:
    ok = st.button("Find Most Similar Invoice", use_container_width=True, on_click=runit)
