FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-dev
RUN apt-get install -y curl

WORKDIR /workspaces
COPY ./pyproject.toml ./poetry.lock /workspaces/

RUN pip install --no-cache-dir poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev --no-interaction --no-ansi