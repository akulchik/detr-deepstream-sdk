DETR_CHECKPOINT=facebook/detr-resnet-101


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
	git clone https://huggingface.co/${DETR_CHECKPOINT} data/model
