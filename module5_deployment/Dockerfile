FROM python:3.11.9-slim-bookworm

RUN apt-get -y update && \
    apt-get install --no-install-recommends -y git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY ./requirements.txt .

RUN pip install --no-cache-dir \
    mlflow==2.11.3 \
    numpy==1.26.4 \
    pandas==2.2.1 \
    scikit-learn==1.3.0

ARG USERNAME=user
ARG USER_ID=1000
ARG GROUP_ID=1000

RUN addgroup --gid $GROUP_ID $USERNAME && \
    adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USERNAME

USER $USERNAME

COPY ./ ./

CMD mlflow models serve -m $(pwd)/models/hgb --no-conda -h 0.0.0.0 -p 5002