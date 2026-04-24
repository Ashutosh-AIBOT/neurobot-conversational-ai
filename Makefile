.PHONY: install streamlit api benchmark benchmark-live compile

install:
	python3 -m pip install -r requirements.txt

streamlit:
	streamlit run app.py

api:
	uvicorn api.main:app --reload

benchmark:
	python3 scripts/run_benchmark.py

benchmark-live:
	python3 scripts/run_benchmark.py --live

compile:
	python3 -m compileall .
