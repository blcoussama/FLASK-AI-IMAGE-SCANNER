FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV FLASK_APP=app.py
EXPOSE 7860
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=7860"]