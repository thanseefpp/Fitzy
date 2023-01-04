FROM python:3.7
WORKDIR usr/src/flask
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# WORKDIR usr/src/flask
# COPY . /app/
# RUN pip install -r requirements.txt
# CMD ["uwsgi","app.ini"]