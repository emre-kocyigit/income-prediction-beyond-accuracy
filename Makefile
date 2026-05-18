.PHONY: install train test serve clean

install:
	pip install -r requirements.txt

train:
	PYTHONPATH=. python scripts/train_and_save_models.py

test:
	PYTHONPATH=. pytest tests/ -v

serve:
	PYTHONPATH=. uvicorn app.main:app --reload --port 8000

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete