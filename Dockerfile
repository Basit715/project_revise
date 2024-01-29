FROM python:3.8-slim-buster
WORKDIR /service
COPY requirements.txt .
COPY . ./
RUN pip install -r requirements.txt
ENTRYPOINT ["PYTHON3", "APP.PY"] 