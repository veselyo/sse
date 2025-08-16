.PHONY: install front_crash aeris clean

install:
	pip install -r requirements.txt --no-cache-dir

front_crash:
	python3 -m front_crash.main

aeris:
	python3 -m aeris.main

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +