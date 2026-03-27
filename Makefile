.PHONY: lint test test-paths test-gpu build build-cuda12 build-cuda11 test-docker \
       verify-weights test-infer-cuda12 test-infer-cuda11

lint:
	flake8 . --count --select=E9,F402,F6,F7,F5,F8,F9 --show-source --statistics --exclude=nibio_inference,nibio_sparsify,bash_helpers,big_table_creation,visualization,scripts/oracle
	mypy torch_points3d

test:
	python -m pytest tests/ -v --ignore=tests/test_inference.py -x

test-paths:
	python -m pytest tests/test_paths.py -v

test-gpu:
	python -m pytest tests/ -v

build: build-cuda11

build-cuda12:
	docker build -f docker/Dockerfile.cuda12 -t segmentanytree:cuda12 -t segmentanytree:latest .

build-cuda11:
	docker build -f docker/Dockerfile.cuda11 -t segmentanytree:cuda11 .

test-docker:
	bash tests/test_docker_build.sh

infer-parallel:
	bash scripts/run_inference_parallel.sh $(INPUT) $(OUTPUT) $(GPUS)

train-ddp:
	torchrun --nproc_per_node=$(GPUS) train.py task=panoptic data=panoptic/treeins \
		models=panoptic/area4_ablation_3heads \
		model_name=PointGroup-PAPER training=treeins job_name=ddp_run

# ---- Weight verification ----
verify-weights:
	@MODEL_SIZE=$$(stat -c%s model_file/PointGroup-PAPER.pt 2>/dev/null || echo 0); \
	if [ "$$MODEL_SIZE" -lt 1000 ]; then \
		echo "ERROR: model_file/PointGroup-PAPER.pt is a Git LFS pointer ($$MODEL_SIZE bytes)."; \
		echo "Fix with: git lfs pull --include=model_file/PointGroup-PAPER.pt"; \
		echo "Or:       curl -L -o model_file/PointGroup-PAPER.pt https://github.com/SmartForest-no/SegmentAnyTree/raw/main/model_file/PointGroup-PAPER.pt"; \
		exit 1; \
	fi; \
	echo "Model weights OK: $$MODEL_SIZE bytes"

# ---- Download FOR-instance sample data ----
DATA_DIR ?= $(HOME)/segmentanytree

download-forinstance:
	mkdir -p $(DATA_DIR)/for-instance
	curl -L -o $(DATA_DIR)/for-instance/FORinstance_dataset.zip \
		"https://zenodo.org/records/8287792/files/FORinstance_dataset.zip?download=1"
	cd $(DATA_DIR)/for-instance && unzip -o FORinstance_dataset.zip

# ---- Test inference with FOR-instance RMIT (smallest file) ----
test-infer-cuda12: verify-weights
	mkdir -p $(DATA_DIR)/input $(DATA_DIR)/output_cuda12
	cp $(DATA_DIR)/for-instance/RMIT/test.las $(DATA_DIR)/input/
	docker run --gpus all --rm \
		-v $(DATA_DIR)/input:/data/input \
		-v $(DATA_DIR)/output_cuda12:/data/output \
		segmentanytree:cuda12 \
		bash scripts/run_inference.sh /data/input /data/output true

test-infer-cuda11: verify-weights
	mkdir -p $(DATA_DIR)/input $(DATA_DIR)/output_cuda11
	cp $(DATA_DIR)/for-instance/RMIT/test.las $(DATA_DIR)/input/
	docker run --gpus all --rm \
		-v $(DATA_DIR)/input:/data/input \
		-v $(DATA_DIR)/output_cuda11:/data/output \
		segmentanytree:cuda11 \
		bash scripts/run_inference.sh /data/input /data/output true
