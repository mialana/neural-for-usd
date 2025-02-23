freeze:
	pip list --format=freeze > requirements.txt

build:
	sh ez_build.sh

run:
	sh ez_run.sh