.PHONY: lint test test-paths test-gpu build test-docker

lint:
	flake8 . --count --select=E9,F402,F6,F7,F5,F8,F9 --show-source --statistics --exclude=nibio_inference,nibio_sparsify,bash_helpers,big_table_creation,visualization,scripts/oracle
	mypy torch_points3d

test:
	python -m pytest tests/ -v --ignore=tests/test_inference.py -x

test-paths:
	python -m pytest tests/test_paths.py -v

test-gpu:
	python -m pytest tests/ -v

build:
	docker build -t segmentanytree:latest .

test-docker:
	bash tests/test_docker_build.sh
