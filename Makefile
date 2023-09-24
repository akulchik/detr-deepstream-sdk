# Project structure
BUILD_DIR=build
RELEASE_DIR=${BUILD_DIR}/release
SRC_DIR=src
CXX_SRC_DIR=${SRC_DIR}/cpp
BBOX_SRC_DIR=${CXX_SRC_DIR}/bbox
DETR_CHECKPOINT=facebook/detr-resnet-101
DETR_DIR=data/model
DETR_ONNX_DIR=${DETR_DIR}/onnx
DETR_PT_DIR=${DETR_DIR}/pt

# NVIDIA CUDA configuration
CUDA_VERSION=12.1
CUDA_PATH=/usr/local/cuda-${CUDA_VERSION}
CUDA_INCLUDES=${CUDA_PATH}/include

# NVIDIA DeepStream SDK configuration
DEEPSTREAM_VERSION=6.3
DEEPSTREAM_PATH=/opt/nvidia/deepstream/deepstream-${DEEPSTREAM_VERSION}
DEEPSTREAM_INCLUDES=${DEEPSTREAM_PATH}/sources/includes

# NVIDIA libraries
NV_LIBS:= -lnvinfer -lnvparsers -L${CUDA_PATH}/lib64
NV_LIB_FLAGS:= -Wl,--start-group ${NV_LIBS} -Wl,--end-group

# C++ host compiler flags (GCC) and dependencies
CXX_STANDARD=20
CXX_FLAGS= -std=c++${CXX_STANDARD} -fPIC -shared -O3 -Wall -Werror
CXX_FLAGS+= -I${DEEPSTREAM_INCLUDES} -I${CUDA_INCLUDES}

.PHONY: all
all: py-dependencies detr-download detr-pt-to-onnx

.PHONY: git-lfs
	git lfs install

.PHONY: py-venv
py-venv:
	. venv/bin/activate

.PHONY: py-dependencies
py-dependencies: py-venv
	pip install -r requirements.txt

.PHONY: detr-download
detr-download: git-lfs
	git clone https://huggingface.co/${DETR_CHECKPOINT} ${DETR_PT_DIR}

.PHONY: detr-pt-to-onnx
detr-pt-to-onnx: ${DETR_PT_DIR}/pytorch_model.bin
	optimum-cli export onnx --model ${DETR_PT_DIR} --task object-detection ${DETR_ONNX_DIR}/

libdetr-bbox.so: ${BBOX_SRC_DIR}/nvds_parse_pred_boxes.h ${BBOX_SRC_DIR}/nvds_parse_pred_boxes.cpp
	g++ -o ${RELEASE_DIR}/$@ $< ${CXX_FLAGS} ${NV_LIB_FLAGS}

.PHONY: run-deepstream-app
run-deepstream-app: detr-pt-to-onnx
	deepstream-app -c ./configs/config_infer_detr_resnet_101.txt

.PHONY: clean
clean:
	rm -rf ${BUILD_DIR}
