FROM nvcr.io/nvidia/deepstream:6.1-triton

ENV GIT_SSL_NO_VERIFY=1

# RUN sh docker_python_setup.sh
# RUN update-alternatives --set python3 /usr/bin/python3.8
RUN python3 --version
RUN apt update
RUN apt install --fix-broken -y
RUN apt -y install python3-gi python3-gst-1.0 python-gi-dev git python3 python3-pip cmake g++ build-essential \
  libglib2.0-dev python3-dev python3.8-dev libglib2.0-dev-bin python-gi-dev libtool m4 autoconf automake

RUN cd /opt/nvidia/deepstream/deepstream-6.1/sources/apps && \
    git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git
RUN cd /opt/nvidia/deepstream/deepstream-6.1/sources/apps/deepstream_python_apps && \
    git submodule update --init
RUN cd /opt/nvidia/deepstream/deepstream-6.1/sources/apps/deepstream_python_apps/3rdparty/gst-python/ && \
   ./autogen.sh && \
   make && \
   make install

RUN pip3 install --upgrade pip
RUN cd /opt/nvidia/deepstream/deepstream-6.1/sources/apps/deepstream_python_apps/bindings && \
    mkdir build && \
    cd build && \
    cmake -DPYTHON_MAJOR_VERSION=3 -DPYTHON_MINOR_VERSION=8 -DPIP_PLATFORM=linux_x86_64 -DDS_PATH=/opt/nvidia/deepstream/deepstream-6.1 .. && \
    make && \
    pip3 install pyds-1.1.4-py3-none-linux_x86_64.whl
RUN cd /opt/nvidia/deepstream/deepstream-6.1/sources/apps/deepstream_python_apps && \
    mv apps/* ./
RUN pip3 install --upgrade pip
RUN pip3 install numpy opencv-python

# RTSP
RUN apt update && \
    apt install -y python3-gi python3-dev python3-gst-1.0
RUN apt update && \
    apt install -y libgstrtspserver-1.0-0 gstreamer1.0-rtsp && \
    apt install -y libgirepository1.0-dev && \
    apt-get install -y gobject-introspection gir1.2-gst-rtsp-server-1.0

# DEVELOPMENT TOOLS
RUN apt install -y ipython3 graphviz

RUN apt update && \
    apt install -y valgrind

RUN pip3 install kafka-python msgpack pymongo confluent-kafka numba progressbar2 scikit-image

RUN pip3 install -U opencv-python

RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt update && apt install -y ffmpeg libc6 libc6-dev libnuma1 libnuma-dev libx264-dev

RUN pip install pika requests fastapi uvicorn 

# Custom plugin
# COPY gst-nvinfer /gst-nvinfer
# WORKDIR /gst-nvinfer
# RUN CUDA_VER=11.3 make && CUDA_VER=11.3 make install