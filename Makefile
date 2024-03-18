
format:
	pdm run black src
	pdm run black hypertune.py
	pdm run isort src
	pdm run isort hypertune.py

lint:
	pdm run ruff check src --fix
	pdm run ruff check hypertune.py --fix

run:
	pdm run python hypertune.py
