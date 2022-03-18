format:
	black .
	isort .

backend:
	python src/app.py

frontend:
	streamlit run src/frontend.py