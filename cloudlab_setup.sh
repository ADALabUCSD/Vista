#!/usr/bin/env bash

# Install python and PIP
sudo apt-get update
sudo apt-get install build-essential -y
sudo apt-get install --reinstall systemd -y
sudo apt-get install curl python-dev -y
curl -O https://bootstrap.pypa.io/get-pip.py && sudo python get-pip.py && rm get-pip.py


sudo pip install -r $(dirname "$0")/requirements.txt

