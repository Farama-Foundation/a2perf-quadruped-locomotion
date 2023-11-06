FROM ubuntu:20.04

ARG APT_COMMAND="apt-get -o Acquire::Retries=3 -y"
ARG USER_ID
ARG USER_GROUP_ID

# Installs basics including add-apt.
RUN ${APT_COMMAND} update && ${APT_COMMAND} install -y --no-install-recommends \
    software-properties-common \
    curl \
    wget

# Install Python 3.7.11 and Pip
RUN ${APT_COMMAND} update && \
    ${APT_COMMAND} install software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    ${APT_COMMAND} update && \
    ${APT_COMMAND} install python3.9 python3.9-distutils python3.9-dev python3.9-venv && \
    ${APT_COMMAND} clean && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

RUN curl -O https://bootstrap.pypa.io/get-pip.py

RUN python3.9 get-pip.py

# Install dependencies
RUN ${APT_COMMAND} update && ${APT_COMMAND} install -y wget \
    sudo \
    unzip \
    curl \
    vim \
    less \
    git \
    libopenmpi-dev \
    g++ \
    ffmpeg \
    libsm6 \
    htop \
    ssh \
    openssh-server \
    libxext6

# Python Packages
#RUN pip install attrs==19.3.0
#RUN pip install gym==0.17.1
#RUN pip install mpi4py==3.0.3
#RUN pip install numpy==1.18.0
#RUN pip install pybullet==3.0.6
#RUN pip install tensorflow==1.15.4
#RUN pip install tensorboard==1.15.0
#RUN pip install typing==3.7.4.1
#RUN pip install stable-baselines==2.10.0
#RUN pip install tqdm
#RUN pip install numba
#RUN pip install quadprog
#RUN pip install inputs
#RUN pip install protobuf==3.20.*
#RUN pip install gin-config
RUN pip install attrs
RUN pip install gymnasium
RUN pip install mpi4py
RUN pip install numpy
RUN pip install pybullet
RUN pip install tensorflow
RUN pip install tensorboard
RUN pip install typing
#RUN pip install stable-baselines
RUN pip install tqdm
RUN pip install numba
RUN pip install quadprog
RUN pip install inputs
RUN pip install protobuf
RUN pip install gin-config

# Enable X11 forwarding for docker
RUN echo "X11UseLocalhost no" >> /etc/ssh/sshd_config && \
    echo "X11DisplayOffset 10" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config

# Add User for safely using mpi4py
RUN groupadd -g $USER_GROUP_ID user_group
RUN groupadd -g 998 docker

RUN useradd -u $USER_ID -g $USER_GROUP_ID -G 998 -ms /bin/bash user

# Granting sudo privileges
RUN usermod -aG sudo user
RUN echo "user:password" | chpasswd
RUN echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN mkdir /home/user/.ssh
RUN chmod 700 /home/user/.ssh
RUN touch /home/user/.ssh/authorized_keys
RUN chmod 600 /home/user/.ssh/authorized_keys
ADD .ssh/id_rsa.pub /home/user/.ssh/authorized_keys


RUN echo "Port 2020" >> /etc/ssh/sshd_config
EXPOSE 2020


USER user

ENTRYPOINT sudo service ssh restart && bash




