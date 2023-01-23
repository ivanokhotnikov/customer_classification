timestamp := $(shell date +%Y%m%d%H%M%S)
export

pre-commit:
	pre-commit run --all-files

compile:
	mkdir -p ./logs/$@
	python ./src/xgb_training_pipeline.py --compile_only &>>./logs/$@/$(timestamp).log

default-run:
	mkdir -p ./logs/$@
	python ./src/xgb_training_pipeline.py --enable_caching &>>./logs/$@/$(timestamp).log

custom-run:
	mkdir -p ./logs/$@
	python ./src/xgb_training_pipeline.py ${params} --enable_caching &>>./logs/$@/$(timestamp).log

grid-tuning-run:
	mkdir -p ./logs/$@
	python ./src/xgb_grid_search_tuning.py &>>./logs/$@/$(timestamp).log

random-tuning-run:
	mkdir -p ./logs/$@
	python ./src/xgb_random_search_tuning.py &>>./logs/$@/$(timestamp).log

unit-tests:
	mkdir -p ./logs/$@
	python -m pytest ./src/tests/ &>>./logs/$@/$(timestamp).log

clean:
	python ./src/clean_vertex.py -all -v` &>>./logs/$@/$(timestamp).log

venv:
	python -m venv .venv &&\
	source ./.venv/Scripts/activate &&\
	python -m pip install --upgrade pip setuptools &&\
	pip install -r configs/requirements.txt &&\
