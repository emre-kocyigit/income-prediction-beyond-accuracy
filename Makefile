.PHONY: install test serve clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

serve:
	uvicorn app.main:app --reload --port 8000

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete