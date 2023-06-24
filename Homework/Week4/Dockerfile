#FROM python:3.10.0-slim
FROM svizor/zoomcamp-model:mlops-3.10.0-slim

RUN pip install -U pip
RUN pip install pandas
RUN pip install scikit-learn==1.2.2
RUN pip install pyarrow

WORKDIR /app

#COPY [ "Pipfile", "Pipfile.lock", "./" ]
#RUN pipenv shell --system --deploy

COPY [ "starter.py", "./" ]

ENTRYPOINT ["python", "starter.py", "--year", "2022", "--month", "4"]
    