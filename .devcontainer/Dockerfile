FROM ubuntu:20.04

ARG APT_COMMAND="apt-get -o Acquire::Retries=3 -y"

# Installs basics including add-apt.
RUN ${APT_COMMAND} update && ${APT_COMMAND} install -y --no-install-recommends \
        software-properties-common \
        curl \
        less 

# Install Python 3.7.11 and Pip
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.7 python3.7-distutils python3.7-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1

RUN curl -O https://bootstrap.pypa.io/get-pip.py

RUN python3.7 get-pip.py

# Install dependencies
RUN apt-get update && apt-get install -y wget \
    sudo \
    unzip \
    curl \
    vim \
    git \
    libopenmpi-dev \
    g++ \
    ffmpeg \
    libsm6 \
    libxext6

# Python Packages
RUN pip install attrs==19.3.0
RUN pip install gym==0.17.1
RUN pip install mpi4py==3.0.3
RUN pip install numpy==1.18.0
RUN pip install pybullet==3.0.6
RUN pip install tensorflow==1.15.4
RUN pip install tensorboard==1.15.0
RUN pip install typing==3.7.4.1
RUN pip install stable-baselines==2.10.0
RUN pip install tqdm
RUN pip install numba
RUN pip install quadprog
RUN pip install inputs
RUN pip install protobuf==3.20.*
RUN pip install gin-config

# Add User for safely using mpi4py
RUN useradd -m newuser 
RUN echo 'newuser:mypassword' | chpasswd
RUN usermod -aG sudo newuser
USER newuser
WORKDIR /home/newuser
CMD ["bash"]




