timestamp := $(shell date +%Y%m%d%H%M%S)
export

pre-commit:
	pre-commit run --all-files

default-run:
	mkdir -p ./logs/$@
	python ./src/local/xgb_training_pipeline.py ${params} --timestamp=$(timestamp) &>> ./logs/$@/$(timestamp).log

compile-vertex:
	mkdir -p ./logs/$@
	python ./src/vertex/xgb_training_pipeline.py --compile_only &>>./logs/$@/$(timestamp).log

default-run-vertex:
	mkdir -p ./logs/$@
	python ./src/vertex/xgb_training_pipeline.py --enable_caching &>>./logs/$@/$(timestamp).log

custom-run-vertex:
	mkdir -p ./logs/$@
	python ./src/vertex/xgb_training_pipeline.py ${params} --enable_caching &>>./logs/$@/$(timestamp).log

grid-tuning-run-vertex:
	mkdir -p ./logs/$@
	python ./src/vertex/xgb_grid_search_tuning.py &>>./logs/$@/$(timestamp).log

random-tuning-run-vertex:
	mkdir -p ./logs/$@
	python ./src/vertex/xgb_random_search_tuning.py &>>./logs/$@/$(timestamp).log

unit-tests:
	mkdir -p ./logs/$@
	python -m pytest ./src/vertex/tests/ &>>./logs/$@/$(timestamp).log

clean-vertex:
	python ./src/vertex/clean_vertex.py -all -v` &>>./logs/$@/$(timestamp).log

venv-dev:
	python -m venv .venv &&\
	source ./.venv/Scripts/activate &&\
	python -m pip install --upgrade pip setuptools &&\
	pip install -r configs/requirements-dev.txt &&\
