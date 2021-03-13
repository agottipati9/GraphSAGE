#!/bin/bash

apt update
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"
apt update
apt-cache policy docker-ce
apt install -y docker-ce
chmod 666 /var/run/docker.sock

echo "Finished Docker Install"

echo -ne '\n' | add-apt-repository ppa:deadsnakes/ppa
apt update
apt install -y python3.6
apt install -y python3-pip

echo "Installed Python 3.6"

