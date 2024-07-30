# Invoice Doppelganger

**InvoiceDoppelganger** is a tool designed to compare and find similarities between invoices. It leverages advanced text processing and image analysis techniques to identify duplicate or closely related invoices, ensuring accuracy and efficiency in document management.

It's A Program which takes an input invoice in the form of PDF and compares it to a database of existing invoices based on Content and Similarity.

#### Similarity Metrics Used: 
1. Structure or Style of Tables in Invoices
2. PDF Metadata
3. Invoice Number
4. PDF Name
5. Image Similarity
6. Cosine Similarity of all Metrics

### Key Features:
- Upto 99% Similarity Reached
- Uses Less Resources

### This Involves 3 Steps:


## Step-1: Feature Extraction
- Extract Text from PDF using PyPDF2
- Features Extracted:
    **Text, metadata, table styles from html, invoice number, date, pdf-name**
- Analyze layout and structure using table styles
- Add the training data to the database(list of feature vectors)

- #### Libraries Used :
    1. PyPDF2
    2. pdfminer
    3. sklearn
    4. re
    5. io


## Step-2: Calculate Similarity
- Using Cosine Similarity between features that have being extraced between two extracted feature vectors

- Using Image Similarity converting PDF into image and comparing them getting the similarities.

- Combine Both the Similarities and return the result
- #### Libraries Used :
    1. sklearn
    2. imagehash
    3. numpy
    4. pdf2image

## Step-3: Compare with training data
-  Compare the invoice with each training data and get the most similar invoice and return the similarity

## Final Step:
-  Create A Frontend Using Streamlit

## Steps to Run into your local:

1. First Clone the Repository into your local machine using git

    ```bash
    git clone https://github.com/definitelynotchirag/Invoice-Doppelganger
    ```

2. Install the required dependencies using
    ```bash
    pip3 install -r requirements.txt
    ```
3. You are ready to run the Program

4. **Before running make sure to add the test data(testing pdf invoices) and training data to the respective folders**


**There are two ways of running this program:**


### 1. GUI(Streamlit)
- Run the '**frontend.py**' through streamlit
    ```bash
    streamlit run frontend.py
    ```
- Input select the the invoice which you want to predict on.
- Run the '**Find Most Similar Invoice**' Button
- You will get the Most similar invoice as well as the similarity from 0 to 1

### 2. Command-Line(Python)
- Run the '**main.py**' using
    ```bash
    python3 main.py
    ```
- It will display the input invoice, most similar invoice and the similarity score into the shell
