FROM python:3.11

WORKDIR /app

COPY . .

RUN pip install pydantic openai

CMD ["python", "inference.py"]
