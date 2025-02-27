.PHONY: *

freeze:
	pip list --format=freeze > requirements.txt

build: ez_build.sh
	sh ez_build.sh

run: build ez_run.sh
	sh ez_run.sh


SETTINGS2=--camera=MyCam --complexity veryhigh -- --colorCorrectionMode=openColorIO --frames=0:99

high:
	usdrecord $(SETTINGS2) japanesePlaneToy.usda data/japanesePlaneToy/val/r###.png