.PHONY: *

freeze:
	pip list --format=freeze > requirements.txt

build: ez_build.sh
	sh ez_build.sh

run: ez_run.sh
	sh ez_run.sh


SETTINGS1=--camera=MyCam --complexity=veryhigh --colorCorrectionMode=openColorIO --frames=0:99  --enableDomeLightVisibility

SETTINGS2=--camera=MyCam --complexity=veryhigh --frames=0:99  --enableDomeLightVisibility

high:
	usdrecord $(SETTINGS2) japanesePlaneToy.usda data/japanesePlaneToy/val/r###.png