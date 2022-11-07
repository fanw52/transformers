#FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
#LABEL maintainer="Hugging Face"
#LABEL repository="transformers"
#ENV LANG C.UTF-8
#
#RUN apt update && \
#    apt install -y bash \
#                   build-essential \
#                   git \
#                   curl \
#                   ca-certificates \
#                   python3 \
#                   python3-pip && \
#    rm -rf /var/lib/apt/lists
#
#RUN python3 -m pip install --no-cache-dir --upgrade pip && \
#    python3 -m pip install --no-cache-dir mkl torch -i https://pypi.tuna.tsinghua.edu.cn/simple
#
#
#RUN git clone https://github.com/NVIDIA/apex
#RUN cd apex && \
#    python3 setup.py install && \
#    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
#
#WORKDIR /transformers
#COPY . /transformers
#RUN  python3 -m pip install --no-cache-dir .
#
#RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
#CMD ["/bin/bash"]


FROM transformers:v1.0.0
COPY . /transformers
RUN apt-get update && apt-get install -y vim
RUN pip install sklearn seqeval torchvision==0.8.2 jsonlines -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install detectron2==0.3