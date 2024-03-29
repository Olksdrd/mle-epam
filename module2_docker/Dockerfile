FROM ubuntu:jammy

# Configures time inside the container
ENV TZ=Europe/Kyiv
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get -y update && apt-get -y install \
    git \
    python3-pip=22.0.2+dfsg-1ubuntu0.4 \
    tzdata
WORKDIR /code

COPY ./requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

ARG USERNAME=user
ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID $USERNAME && \
    adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USERNAME

USER $USERNAME

COPY ./ ./