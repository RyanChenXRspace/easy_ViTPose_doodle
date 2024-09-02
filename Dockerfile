FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-dev

WORKDIR /app


RUN pip install --upgrade pip \
    && pip install numpy pandas matplotlib

RUN pip install json_tricks
RUN pip install munkres


COPY ./requirements.txt /app
COPY ./requirements_gpu.txt /app
