pre-commit:
	pre-commit run --all-files

default-run:
	python ./src/local/xgb_training_pipeline.py ${params}

compile-vertex:
	python ./src/vertex/xgb_training_pipeline.py --compile_only

default-run-vertex:
	python ./src/vertex/xgb_training_pipeline.py --enable_caching

custom-run-vertex:
	python ./src/vertex/xgb_training_pipeline.py ${params} --enable_caching

grid-tuning-run-vertex:
	python ./src/vertex/xgb_grid_search_tuning.py

random-tuning-run-vertex:
	python ./src/vertex/xgb_random_search_tuning.py

unit-tests:
	python -m pytest ./src/vertex/tests/

clean-vertex:
	python ./src/vertex/clean_vertex.py -all -v

venv-dev:
	python -m venv .venv &&\
	source ./.venv/Scripts/activate &&\
	python -m pip install --upgrade pip setuptools &&\
	pip install -r configs/requirements-dev.txt &&\
