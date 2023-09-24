FROM nvcr.io/nvidia/deepstream:6.3-triton-multiarch

# Taken from https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_docker_containers.html
# To get video driver libraries at runtime (libnvidia-encode.so/libnvcuvid.so)
ENV NVIDIA_DRIVER_CAPABILITIES $NVIDIA_DRIVER_CAPABILITIES,video
