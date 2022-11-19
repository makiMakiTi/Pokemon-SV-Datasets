FROM python:3

RUN apt-get update
RUN pip install --upgrade pip
RUN python -m pip install jupyterlab

RUN apt-get install -y ffmpeg

# utils
RUN pip install pillow \
                matplotlib \
                numpy \
                loguru \
                opencv-python

# coco split
RUN pip install sklearn \
                funcy \
                argparse \
                scikit-multilearn \
                scikit-learn



