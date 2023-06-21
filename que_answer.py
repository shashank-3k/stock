import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os
import openai
import time
from PIL import Image
#import flair
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from nltk.tokenize import sent_tokenize
import PyPDF2

# set your OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = "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"




def qna_main():
    st.title("Auditor Report Summarizer")
    stock = st.selectbox("Select a stock", ["AAPL", "AMZN", "MSFT", "GOOG", "TSLA", "NVDA", "COST", "ADBE","META", "WMT"])
    file_loc = "./reports/"
    pdf_file = file_loc + stock.lower() + ".txt"
    with open(pdf_file, encoding='iso-8859-1') as f:
          state_of_the_union = f.read()
    
    # if stock == 'AAPL':
    #     with open(r"C:\Users\pande\Downloads\aaple.txt", encoding='iso-8859-1') as f:
    #       state_of_the_union = f.read()
    # elif stock=="AMZN": 
    #     with open(r"C:\Users\pande\Downloads\amzn.txt", encoding='iso-8859-1') as f:
    #       state_of_the_union = f.read()
        
    # elif stock=="GOOG":
    #     with open(r"C:\Users\pande\Downloads\googl.txt", encoding='iso-8859-1') as f:
    #       state_of_the_union = f.read() 
          
    # elif stock=="MSFT":
    #     with open(r"C:\Users\pande\Downloads\msft.txt", encoding='iso-8859-1') as f:
    #       state_of_the_union = f.read()
          
    # elif stock=="NVDA":
    #     with open(r"C:\Users\pande\Downloads\nvda .txt", encoding='iso-8859-1') as f:
    #       state_of_the_union = f.read()
          
    # elif stock=="TSLA":
    #     with open(r"C:\Users\pande\Downloads\tsla.txt", encoding='iso-8859-1') as f:
    #       state_of_the_union = f.read()
          
    # elif stock=="WMT":
    #     with open(r"C:\Users\pande\Downloads\wmt.txt", encoding='iso-8859-1') as f:
    #       state_of_the_union = f.read()
          
    # elif stock=="ADBE":
    #     with open(r"C:\Users\pande\Downloads\adbe.txt", encoding='iso-8859-1') as f:
    #       state_of_the_union = f.read()
          
    # elif stock=="COST":
    #     with open(r"C:\Users\pande\Downloads\cost.txt", encoding='iso-8859-1') as f:
    #       state_of_the_union = f.read()
          
    # elif stock=="META":
    #     with open(R"C:\Users\pande\Downloads\meta.txt", encoding='iso-8859-1') as f:
    #       state_of_the_union = f.read()
        
         

# Close the PDF file
    

        # pdf_file = open(r"C:\Users\pande\Downloads\aaple.pdf", 'rb')
        # pdf_reader = PyPDF2.PdfReader(pdf_file)

        # text = ""
        # for page in pdf_reader.pages:
        #     text += page.extract_text()

        # pdf_file.close()

            
   
    
    

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(state_of_the_union)

    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))]).as_retriever()
    selected_question = st.selectbox("Select a question", [
    "What is the scope of the audit? What areas of the company's financial statements were audited?",
    "Were any significant accounting policies or estimates changed during the year? If so, how did the auditor evaluate these changes?",
    "What were the auditor's overall findings and conclusions? Were there any material misstatements or weaknesses in internal controls?",
    "Did the auditor identify any fraud or other irregularities? If so, how did they respond to these findings?",
    "What were the auditor's recommendations for improving the company's financial reporting processes?",
    "Did the auditor provide an opinion on the company's financial statements? If so, what was the opinion?",
    "Were there any disagreements between the auditor and management? If so, how were these disagreements resolved?",
    "Did the auditor provide any additional assurance services beyond the standard audit? If so, what were the results of these services?",
    "What was the auditor's fee for the audit and related services?",
    "What is the auditor's track record in terms of identifying financial reporting issues or weaknesses in internal controls at other companies?"
])
    st.write("Selected question:", selected_question)


    #query = st.text_input('Enter your question here:')
    docs = docsearch.get_relevant_documents(selected_question) 
    # Define the submit button
    # with st.spinner('Wait for it...'):
    #     time.sleep(5)
    #st.success('Done!')
    if st.button('Submit',key=1):
        with st.spinner('Wait for it...'):
           time.sleep(5)
        st.success('Done!')
        chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
        answers = chain.run(input_documents=docs, question=selected_question)
        st.header('Answer:')
        st.markdown(answers)
        st.header('Summary:')

        st.markdown(docs)

        # chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
        # answers = chain.run(input_documents=docs, question=selected_question)
        # st.header('Answer:')
        # st.markdown(answers)
    sentiment_model = flair.models.TextClassifier.load('en-sentiment')
    text = flair.data.Sentence(answers)
    sentiment_model.predict(text)
    # color = st.color_picker('#00f94f')
    st.write("",text) 


if __name__ == '__main__':
    qna_main()
