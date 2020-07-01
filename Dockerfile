ARG tag=20.06
FROM nvcr.io/nvidia/pytorch:${tag}-py3
ARG tag

ENV CLIENT_DIR=/opt/tensorrtserver/client
RUN mkdir -p ${CLIENT_DIR} && \
      RELEASE=$(curl -s https://raw.githubusercontent.com/NVIDIA/tensorrt-inference-server/r${tag}-v1/VERSION) && \
      wget -O ${CLIENT_DIR}/clients.tar.gz https://github.com/NVIDIA/tensorrt-inference-server/releases/download/v${RELEASE}/v${RELEASE}_ubuntu1804.clients.tar.gz && \
      cd ${CLIENT_DIR} && \
      tar xzf clients.tar.gz && \
      pip install --upgrade ${CLIENT_DIR}/python/tensorrtserver-*.whl