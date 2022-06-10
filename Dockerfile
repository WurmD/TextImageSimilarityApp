FROM ubuntu:20.04

RUN apt update \
    && apt install -y apt-utils wget \
    && apt install -y git python3 python3-pip \
    && DEBIAN_FRONTEND=noninteractive apt install -y libglib2.0-0 libsm6 libxrender1 libxext6 \
    && pip3 install -U pip setuptools wheel \
    && pip3 install argparse falcon gunicorn nltk omegaconf opencv-python==4.2.0.34 \
    && pip3 install scikit-image sklearn spacy torch wget \
    && python3 -m spacy download en_core_web_sm \
    && pip3 install matplotlib \
    && pip install cython \
    && pip install git+https://github.com/lucasb-eyer/pydensecrf.git
COPY  ./ /TextImageSimilarityApp
WORKDIR /TextImageSimilarityApp
RUN wget 'https://www.dropbox.com/s/icpi6hqkendxk0m/deeplabv2_resnet101_msc-cocostuff164k-100000.pth?raw=1' -O deeplabpytorch/deeplabv2_resnet101_msc-cocostuff164k-100000.pth
