FROM ubuntu:latest

# Ubuntu stuff.
RUN apt update && apt upgrade -y && \
    apt install -y software-properties-common libpng-dev libjpeg-dev libtiff-dev wget curl libgl1-mesa-glx && \
    TZ="Russia/Moscow" apt-get install -y tzdata

# Install python.
RUN add-apt-repository ppa:deadsnakes/ppa && apt install -y python3.9 python3.9-dev python3.9-distutils \
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.9 get-pip.py \
    && python3.9 -m pip install --upgrade pip

# App stuff.
COPY backend/requirements.txt .
RUN python3.9 -m pip install -r requirements.txt
RUN mkdir -p /usr/src/app
COPY . /usr/src/app
WORKDIR /usr/src/app

# Download SAM weights.
RUN python3.9 -m pip install gdown
RUN mkdir backend/weights
RUN gdown https://drive.google.com/file/d/1XyumF6_fdAxFmxpFcmPf-q84LU_22EMC/view?usp=sharing --fuzzy -O backend/weights/model.pth

EXPOSE 80
ENV PYTHONPATH=backend/$PYTHONPATH
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]