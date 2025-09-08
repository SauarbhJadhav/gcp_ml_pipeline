FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY holtwinter_model.py .
CMD ["python", "holtwinter_model.py"]