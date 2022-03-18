setup:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

format:
	black .
	isort .

backend:
	python src/app.py

frontend:
	streamlit run src/frontend.py