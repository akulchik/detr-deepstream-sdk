DETR_CHECKPOINT=facebook/detr-resnet-101
DETR_DIR=data/model
DETR_ONNX_DIR=${DETR_DIR}/onnx
DETR_PT_DIR=${DETR_DIR}/pt


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
