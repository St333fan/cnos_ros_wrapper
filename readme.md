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

To start the ros wrapper just run 

```
docker run -it --runtime nvidia --privileged -e DISPLAY=${DISPLAY}  -e NVIDIA_DRIVER_CAPABILITIES=all -v /home/v4r/demo/object_detectors/cnos/:/code -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/v4r/demo/object_detectors/cnos/torch_cache:/root/.cache/torch --net host --shm-size=2gb --rm cnos
```
