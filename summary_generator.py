import PyPDF2
import re
import nltk
import networkx as nx
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import streamlit as st

nltk.download('stopwords')


stock_list = ["AAPL", "AMZN", "MSFT", "GOOG", "TSLA", "NVDA", "COST", "ADBE","META"]
file_loc = "./reports/"

def preprocess_text(text):
    sentences = sent_tokenize(text)
    words = [word_tokenize(sentence.lower()) for sentence in sentences]

    stop_words = set(stopwords.words('english'))
    words = [[word for word in sentence if word not in stop_words and word.isalnum()] for sentence in words]

    return words

def create_graph(words):
    graph = nx.DiGraph()

    for sentence in words:
        for word in sentence:
            if not graph.has_node(word):
                graph.add_node(word)

    for i in range(len(words)):
        for j in range(i+1, len(words)):
            for word1 in words[i]:
                for word2 in words[j]:
                    if graph.has_node(word1) and graph.has_node(word2):
                        if not graph.has_edge(word1, word2):
                            graph.add_edge(word1, word2)

    return graph

def summarize(text, num_sentences=4):
    words = preprocess_text(text)
    graph = create_graph(words)
    scores = nx.pagerank(graph)
    sentences = sent_tokenize(text)
    sentences_scores = [(sentences[i], sum(scores.get(word, 0) for word in words[i])) for i in range(len(sentences))]
    sentences_scores = sorted(sentences_scores, key=lambda x: x[1], reverse=True)
    summary = [sentences_scores[i][0] for i in range(min(num_sentences, len(sentences_scores)))]
    return ' '.join(summary)

def summary_main():
    st.title("Auditor Report Summarizer")

    stock = st.selectbox("Select a stock", stock_list)

    pdf_file = file_loc + stock.lower() + ".pdf"
    
    pdf_reader = PyPDF2.PdfReader(open(pdf_file, 'rb'))
    num_pages = len(pdf_reader.pages)

    start_page = st.number_input("Enter the start page number", min_value=1, max_value=num_pages, value=1, step=1)
    end_page = st.number_input("Enter the end page number", min_value=1, max_value=num_pages, value=num_pages, step=1)

    if st.button("Submit"):
        if start_page > end_page:
            st.error("Invalid page range! Please make sure the start page is before the end page.")
            return

        with st.spinner("Generating summaries..."):
            summaries = []
            for i in range(start_page - 1, end_page):
                page = pdf_reader.pages[i]
                page_text = page.extract_text()
                page_text = re.sub(r'\s+', ' ', page_text)
                page_text = page_text.strip()
                summary = summarize(page_text, num_sentences=4)
                summaries.append(summary)

        st.write(f"Summary of pages {start_page}-{end_page} for {stock}:")
        for i, summary in enumerate(summaries):
            page_number = i + start_page
            st.markdown(f'<p style="text-align: justify;"><strong>Page {page_number}: </strong>{summary}</p>', unsafe_allow_html=True)





