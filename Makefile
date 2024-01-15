
format:
	pdm run black src
	pdm run isort src

lint:
	pdm run ruff src --fix
