FROM python:3.11.9-bookworm

RUN apt-get -y update && apt-get -y install git

WORKDIR /code

COPY ./requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./ ./

ENTRYPOINT ["/bin/bash"]