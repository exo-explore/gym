# Use an NVIDIA CUDA 12.4 development base image on Ubuntu 22.04
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install required packages, including OpenSSH server, and add the deadsnakes PPA for Python 3.13
RUN apt-get update && \
    apt-get install -y software-properties-common git curl openssh-server tmux neovim && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.13 python3.13-venv python3.13-dev libdrm-dev && \
    rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.13
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.13

# Clone your repository into /opt/exogym
RUN git clone https://github.com/exo-explore/exo.git /opt/exogym

# Set the working directory
WORKDIR /opt/exogym

# Create a virtual environment and install your package in editable mode
RUN python3.13 -m venv .venv && \
    . .venv/bin/activate && \
    pip install -e .

# Configure SSH server
RUN mkdir -p /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    echo 'UsePAM no' >> /etc/ssh/sshd_config

# Create a bash profile to automatically start tmux and activate venv on login
RUN echo 'if [ -z "$TMUX" ]; then\n  tmux new-session -s dev "cd /opt/exogym && source .venv/bin/activate && bash"\nfi' > /root/.bash_profile && \
    echo 'source /root/.bash_profile' >> /root/.bashrc

# Expose SSH port
EXPOSE 22

# Start SSH service and then execute the default command
CMD ["/usr/sbin/sshd", "-D"]
