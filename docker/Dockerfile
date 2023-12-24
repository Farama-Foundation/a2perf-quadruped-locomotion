ARG BASE_IMAGE="nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04"
FROM $BASE_IMAGE

ARG APT_COMMAND="apt-get -o Acquire::Retries=3 -y"
ARG USER_ID
ARG USER_GROUP_ID

ENV DEBIAN_FRONTEND=noninteractive

RUN ${APT_COMMAND} update &&  \
    ${APT_COMMAND} upgrade &&  \
    ${APT_COMMAND} install -y wget  \
    sudo \
    ssh \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*


# Install MIniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ./Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm ./Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/opt/conda/bin:${PATH}"
RUN conda create -n py39 python=3.9 -y
ENV CONDA_DEFAULT_ENV=py39
ENV PATH="/opt/conda/envs/py39/bin:/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

RUN conda init bash
RUN echo "conda activate py39" >> ~/.bashrc


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


RUN conda install -c conda-forge openmpi -y
RUN pip install tensorflow
RUN conda install -c conda-forge mpi4py -y
RUN conda install -c conda-forge pybullet -y


# need libGL.so.1 for pybullet
RUN ${APT_COMMAND} update && \
    ${APT_COMMAND} install -y libgl1-mesa-glx ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

RUN echo "Port 2020" >> /etc/ssh/sshd_config
ADD .ssh/id_rsa.pub /home/user/.ssh/authorized_keys
EXPOSE 2020



ENTRYPOINT service ssh restart && bash




