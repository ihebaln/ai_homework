.PHONY: setup test train eval clean 
 
setup: 
pip install -r requirements.txt 2>nul || true 
pip install pytest pytest-cov 
 
test: 
pytest tests/ -v --cov=src --cov-report=term 
 
train: 
python pipelines\train.py 
 
eval: 
python pipelines\evaluate.py 
 
clean: 
if exist __pycache__ rmdir /s /q __pycache__ 
if exist .pytest_cache rmdir /s /q .pytest_cache 
if exist .coverage del .coverage 
