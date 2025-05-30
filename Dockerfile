FROM python:3.11-slim-bullseye

COPY . /app

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "app.py"]

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
