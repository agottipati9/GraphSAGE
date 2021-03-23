FROM tensorflow/tensorflow:1.3.0

RUN apt update && apt install nano python-tk -y
RUN pip install Cython -y
RUN pip install networkx==1.11 matplotlib Ripser seaborn -y
RUN rm /notebooks/*

COPY . /notebooks
