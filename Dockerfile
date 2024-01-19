ARG BASE_IMG
FROM $BASE_IMG

USER root

RUN apt update && apt install -y curl
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y nodejs

RUN pip install --upgrade -v \
  "datasets" \
  "ipywidgets" \
  "jupyter" \
  "jupyterlab-git" \
  "jupyterlab>=4.0.0" \
  "matplotlib" \
  "pip" \
  "requests" \
  "sentencepiece" \
  "tensorboard" \
  "tqdm==4.62.2" \
  "transformers[torch]" \
  && rm -rf ~/.cache/pip/*
RUN apt update && apt install -y git

RUN if ! id jovyan >/dev/null 2>&1; then \
  useradd -m -u 1000 -g 100 -s /bin/bash -d /home/jovyan jovyan; \
fi
RUN apt install -y sudo \
  && echo "jovyan ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers \
  && usermod -a -G root jovyan \
  && mkdir -p /home/jovyan/.local && mkdir -p /home/jovyan/.jupyter \
  && chown -R 1000:100 /home/jovyan/.local && chown -R 1000:100 /home/jovyan/.jupyter

ENV SHELL=/bin/bash
ENV JUPYTER_DATA_DIR=/home/jovyan/.local
USER jovyan
WORKDIR /home/jovyan
