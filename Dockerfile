FROM python:3.10.6
COPY requirements_fastapi.txt /requirements_fastapi.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements_fastapi.txt
COPY gravitw /code/gravitw
WORKDIR /code
CMD uvicorn gravitw.api.fast:app --host 0.0.0.0 --port $PORT
