
# app.py
import streamlit as st

from PIL import Image


from Financial_Ratios import financial_main
from stock_prediction import stock_main
from summary_generator import summary_main

#####################Sidebar##################

lst = ['Balance Sheet','Income Statement','Cash Flow', 'Dividends', 'Volume', 'Stock Splits', 'Job_Value', 'Labour_values', 'Manufacturing_Value', 'CPI_Value', 'inflation_Value', 'Housing_value']
s = ''
for i in lst:
    s += "- " + i + "\n"

url="3kt.png"
image = Image.open(url)
st.sidebar.image(image, caption='3k Technologies – Smart Technology Solutions')
st.sidebar.markdown("Forecasting Time Series Data - Stock Price ")
st.sidebar.title("For Details on the Stock Data.")
st.sidebar.markdown(s)

##################################################




def about():
    st.title("About Page")
    st.write("This is the About Page.")

def contact():
    st.title("Contact Page")
    st.write("You can reach us on the Contact Page.")

# Define your pages and their respective functions
pages = {
    "AI Stock Prediction": stock_main,
    "Financial Ratios": financial_main,
    "Auditor Report Summarizer": summary_main
}

# Create a sidebar navigation menu
selection = st.sidebar.radio("Navigation", list(pages.keys()))

# Run the function corresponding to the selected page
pages[selection]()
