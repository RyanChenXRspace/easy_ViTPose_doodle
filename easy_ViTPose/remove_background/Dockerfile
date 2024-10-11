FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

ENV TZ=Asia/Taipei \
    DEBIAN_FRONTEND=noninteractive

ENV LANG     en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV LC_ALL   en_US.UTF-8

RUN apt-get update
RUN apt-get install tzdata

RUN apt-get install -y wget build-essential git
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y libsm6
RUN apt-get install -y libxrender1
RUN apt-get install -y libxext6
RUN apt-get install -y libegl1-mesa-dev

# Install conda
RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm -rf ~/miniconda3/miniconda.sh

RUN ~/miniconda3/bin/conda init bash
RUN ~/miniconda3/bin/conda init zsh

# nvidia toolkit
#RUN conda install -y nvidia/label/cuda-11.8.0::cuda-toolkit
#RUN conda install -y pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# RUN conda install -y numpy
# RUN conda install -y scikit-image
# RUN conda install -y pillow
# RUN conda install -y ninja

# RUN pip install --no-cache-dir matplotlib opencv-python-headless tensorflow==2.12 kornia dominate trimesh
