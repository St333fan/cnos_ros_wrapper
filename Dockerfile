FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
	# python3-opencv \
    ca-certificates \
    python3-dev \
    git \
    wget \
    sudo \
    build-essential \
    gcc \
    lsb-release \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    freeglut3-dev \
    freeglut3 \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libxext-dev \
    libxt-dev \
    unzip

RUN ln -sv /usr/bin/python3 /usr/bin/python

RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py && \
	rm get-pip.py

RUN python3 -m pip install \
    torch \
    torchvision \
    omegaconf \
    torchmetrics==0.10.3 \
    fvcore \
    iopath \
    xformers==0.0.18 \
    opencv-python pycocotools \
    matplotlib \
    onnxruntime \
    onnx \
    scipy \
    hydra-colorlog \
    hydra-core \
    pytorch-lightning==1.8.1 \
    pandas \
    ruamel.yaml \
    pyrender \
    wandb \
    distinctipy \
    git+https://github.com/facebookresearch/segment-anything.git \
    ultralytics==8.0.135 \
    gdown \
    unzip \
    scikit-image

###
# Install ros
# add the keys
RUN sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ros-noetic-ros-base \
    ros-noetic-catkin \
    ros-noetic-vision-msgs \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/noetic/setup.bash
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN source ~/.bashrc

# install python dependencies
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    build-essential \
    python3-rosdep \
    python3-catkin-tools \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN sudo rosdep init
RUN rosdep update
RUN mkdir -p /root/catkin_ws/src
RUN /bin/bash -c  '. /opt/ros/noetic/setup.bash; cd /root/catkin_ws; catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so; catkin build'

# clone and build message and service definitions
RUN /bin/bash -c 'cd /root/catkin_ws/src; \
                  git clone https://github.com/v4r-tuwien/object_detector_msgs.git'
RUN /bin/bash -c 'cd /root/catkin_ws/src; \
                  git clone https://gitlab.informatik.uni-bremen.de/robokudo/robokudo_msgs.git'
RUN /bin/bash -c '. /opt/ros/noetic/setup.bash; cd /root/catkin_ws; catkin build'

RUN python3 -m pip install \
    catkin_pkg \
    rospkg

RUN python3 -m pip install \
    git+https://github.com/qboticslabs/ros_numpy.git

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    mesa-utils

WORKDIR /code

CMD ["/bin/bash", "-c", "source /opt/ros/noetic/setup.bash && source /root/catkin_ws/devel/setup.bash && \
    python cnos_ros_wrapper.py dataset_name=ycbv model=cnos_fast model.onboarding_config.rendering_type=pyrender"]


