.PHONY: install data run test app all clean

install:
	pip install -r requirements.txt

data:
	python -m src.data_generator --sqlite data/sem.sqlite

run:
	python -m src.pipeline --report outputs/report.html

test:
	python -m pytest

app:
	streamlit run app.py

all: data run test

clean:
	rm -rf data/*.duckdb data/*.sqlite outputs/sql outputs/*.csv .pytest_cache
