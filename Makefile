.PHONY: *

format: src
	find . -path ./include -prune -o -type f \( -iname '*.h' -o -iname '*.cpp' -o -iname '*.glsl' \) -print | xargs clang-format -i

build: ez_build.sh
	sh ez_build.sh

run: ez_run.sh
	sh ez_run.sh