FROM python:3.11.9-bookworm

RUN apt-get -y update && apt-get -y install git

WORKDIR /code

COPY ./requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ARG USERNAME=user
ARG USER_ID=1000
ARG GROUP_ID=1000

RUN addgroup --gid $GROUP_ID $USERNAME && \
    adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USERNAME

USER $USERNAME

COPY ./ ./

ENTRYPOINT ["/bin/bash"]