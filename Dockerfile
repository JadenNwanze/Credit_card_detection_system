FROM python:3.13-slim-buster

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["python3","-m", "src.main"]