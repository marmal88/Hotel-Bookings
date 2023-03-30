FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .
RUN chmod -R +x ./run.sh

ENTRYPOINT [ "sh", "-c", "./run.sh" ]