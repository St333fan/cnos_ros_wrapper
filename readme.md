Build docker ontainer

```
docker build -t cnos .
```

Set your ros ip stuff in ros_entrypoint.sh


Run docker in terminal

```
docker run -it --runtime nvidia --privileged -e DISPLAY=${DISPLAY}  -e NVIDIA_DRIVER_CAPABILITIES=all -v /home/v4r/demo/object_detectors/cnos/:/code -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/v4r/demo/object_detectors/cnos/torch_cache:/root/.cache/torch --net host --shm-size=2gb --rm cnos /bin/bash 
```

maybe probably run setup.sh if its your first time.

To start the ros wrapper for the bop-ycbv objects just run 

```
docker run -it --runtime nvidia --privileged -e DISPLAY=${DISPLAY}  -e NVIDIA_DRIVER_CAPABILITIES=all -v /home/v4r/demo/object_detectors/cnos/:/code -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/v4r/demo/object_detectors/cnos/torch_cache:/root/.cache/torch --net host --shm-size=2gb --rm cnos
```
To start the ros wrapper for custom objects run the following command (you might have to tweak the parameters like light_intensity, etc. in the 'cnos_custom_ros_wrapper.py' depending on the dataset). This should start the container and provide you with a bash terminal:
```
docker run -it --runtime nvidia --privileged -e DISPLAY=${DISPLAY}  -e NVIDIA_DRIVER_CAPABILITIES=all -v /home/v4r/demo/object_detectors/cnos/:/code -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/v4r/demo/object_detectors/cnos/torch_cache:/root/.cache/torch --net host --shm-size=2gb --rm cnos /bin/bash
```

Afterwards start the custom wrapper with the following command:
```
source ros_entrypoint.sh
python cnos_custom_ros_wrapper.py model=cnos_fast model.onboarding_config.rendering_type=pyrender
```
