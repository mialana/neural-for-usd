.PHONY: *

freeze:
	pip list --format=freeze > requirements.txt

build: ez_build.sh
	sh ez_build.sh

run: build ez_run.sh
	sh ez_run.sh