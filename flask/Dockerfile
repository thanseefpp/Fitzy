FROM python:3.7
WORKDIR usr/src/flask
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -U python-dotenv
COPY . .