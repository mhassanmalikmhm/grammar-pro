# Python 3.13.4 (Wohi version jo Render par chal raha tha)
FROM python:3.13.4-slim

# Container ke andar 'app' folder banao
WORKDIR /app

# Apne saare project files ko copy karo (ai_project samet)
COPY . /app

# Dependencies install karo (Note: File ka naam requirement.txt hai)
RUN pip install --no-cache-dir -r ai_project/requirement.txt

# Cloud Run ko batao ke kis port par sunna hai
ENV PORT 8080

# Application shuru karne ki command (Aapki final working command)
CMD exec gunicorn --bind :$PORT --workers 1 app:app --chdir ai_project