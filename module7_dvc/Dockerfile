FROM python:3.12-slim-bookworm

RUN apt-get -y update && \
    apt-get install --no-install-recommends -y git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY ./requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./ ./

ENTRYPOINT [ "/bin/bash" ]