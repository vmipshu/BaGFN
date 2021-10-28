FROM tensorflow/tensorflow:1.14.0-gpu-py3
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y git --no-install-recommends && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace
RUN git clone https://github.com/vmipshu/BaGFN.git /workspace/BaGFN

WORKDIR /workspace/BaGFN
RUN python -m pip install -r requirements.txt

ENTRYPOINT /bin/bash
