FROM tensorflow/tensorflow:1.3.0

RUN apt update && apt install nano
RUN pip install Cython
RUN pip install networkx==1.11 matplotlib Ripser seaborn
RUN rm /notebooks/*

COPY . /notebooks
