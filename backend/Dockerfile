FROM python:3.9

WORKDIR /usr/src/app

# Ubuntu stuff.
RUN apt update && apt upgrade -y && \
    apt install -y software-properties-common libpng-dev libjpeg-dev libtiff-dev wget curl libgl1-mesa-glx && \
    TZ="Russia/Moscow" apt-get install -y tzdata

# Download SAM weights.
RUN python3.9 -m pip install gdown
RUN mkdir --parents backend/weights
RUN gdown https://drive.google.com/file/d/1XyumF6_fdAxFmxpFcmPf-q84LU_22EMC/view?usp=sharing --fuzzy -O backend/weights/model.pth

# App stuff.
COPY backend/requirements.txt .
RUN python3.9 -m pip install --upgrade pip && python3.9 -m pip install -U -r requirements.txt
RUN mkdir -p /usr/src/app
COPY . /usr/src/app

EXPOSE 80
ENV PYTHONPATH=backend/$PYTHONPATH
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]