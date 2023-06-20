FROM python:3.9

WORKDIR /app

COPY requirements.txt /app

RUN pip install -r requirements.txt

COPY . /app

RUN python3 nltk_loader.py

CMD ["streamlit", "run", "app.py"]

EXPOSE 8501
