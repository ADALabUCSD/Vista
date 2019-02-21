#!/usr/bin/env bash

# Install python and PIP
sudo apt-get update
sudo apt-get install build-essential -y
sudo apt-get install --reinstall systemd -y
sudo apt-get install curl python-dev -y
sudo apt-get remove python-six python-chardet -y
curl -O https://bootstrap.pypa.io/get-pip.py && sudo python get-pip.py && rm get-pip.py

wget https://archive.apache.org/dist/spark/spark-2.2.0/spark-2.2.0-bin-hadoop2.7.tgz
tar -xzf spark-2.2.0-bin-hadoop2.7.tgz

sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:webupd8team/java -y
sudo apt-get update

echo "oracle-java8-installer shared/accepted-oracle-license-v1-1 select true" | sudo debconf-set-selections
sudo apt-get install oracle-java8-installer
sudo apt-get install oracle-java8-set-default

echo 'export JAVA_HOME="/usr/lib/jvm/java-8-oracle"' >> ~/.bashrc
echo 'export SPARK_HOME="$HOME/vista/spark-2.2.0-bin-hadoop2.7"' >> ~/.bashrc
echo 'export PATH="$PATH:$SPARK_HOME/bin"' >> ~/.bashrc
echo 'export PYSPARK_DRIVER_PYTHON="jupyter"' >> ~/.bashrc
echo 'export PYSPARK_DRIVER_PYTHON_OPTS="notebook"' >> ~/.bashrc


sudo pip install -r $(dirname "$0")/requirements.txt

echo "deb https://dl.bintray.com/sbt/debian /" | sudo tee -a /etc/apt/sources.list.d/sbt.list
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 642AC823
sudo apt-get update
sudo apt-get install sbt
