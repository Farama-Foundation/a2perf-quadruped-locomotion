ARG BASE_IMAGE="nvidia/cuda:12.2.0-devel-ubuntu22.04"
FROM $BASE_IMAGE

ARG APT_COMMAND="apt-get -o Acquire::Retries=3 -y"
ARG USER_ID
ARG USER_GROUP_ID

ENV DEBIAN_FRONTEND=noninteractive

RUN ${APT_COMMAND} update && \
    ${APT_COMMAND} install software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    ${APT_COMMAND} update && \
    ${APT_COMMAND} install python3.9 python3.9-distutils python3.9-dev python3.9-venv python3-pip && \
    ${APT_COMMAND} clean && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Install dependencies
RUN ${APT_COMMAND} update && ${APT_COMMAND} install -y  \
    libopenmpi-dev \
    sudo \
    ssh \
    openssh-server \
    libgl1-mesa-glx

# Set CUDA environment variables
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"


RUN python3.9 -m pip install attrs
RUN python3.9 -m pip install gymnasium
RUN python3.9 -m pip install mpi4py
RUN python3.9 -m pip install numpy
RUN python3.9 -m pip install pybullet
RUN python3.9 -m pip install tensorboard
RUN python3.9 -m pip install typing
RUN python3.9 -m pip install tqdm
RUN python3.9 -m pip install numba
RUN python3.9 -m pip install quadprog
RUN python3.9 -m pip install inputs
RUN python3.9 -m pip install protobuf
RUN python3.9 -m pip install gin-config
RUN python3.9 -m pip install pydevd-pycharm
RUN python3.9 -m pip install tensorflow[and-cuda]


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


ENTRYPOINT sudo service ssh restart && bash




