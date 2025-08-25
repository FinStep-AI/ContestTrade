# Phase 1
FROM frolvlad/alpine-miniconda3:latest AS compile-image
WORKDIR /ContestTrade
COPY requirements.txt .
# RUN conda update conda
RUN conda create -n contesttrade python=3.10

RUN echo "conda activate contesttrade" >> ~/.bashrc
ENV PATH /opt/conda/envs/contesttrade/bin:$PATH

RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir -r requirements.txt

# Install conda-pack:
RUN conda install -c conda-forge conda-pack
# Use conda-pack to create a standalone enviornment in /venv:
RUN conda-pack -n contesttrade --ignore-missing-files -o /tmp/env.tar && \
  mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
  rm /tmp/env.tar
RUN /venv/bin/conda-unpack

# Phase 2
FROM frolvlad/alpine-miniconda3:latest AS runtime-image
RUN apk update && apk add bash

WORKDIR /ContestTrade
COPY . .
COPY --from=compile-image /venv /venv

SHELL ["/bin/bash", "-c"]
ENTRYPOINT source /venv/bin/activate && \
           python -m cli.main run
