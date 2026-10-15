# Photos: private, local-first photo search. Everything runs on this Mac (see docs/).
PY := search/.venv/bin/python
DIR ?=

.PHONY: help setup index build dev start stage service unservice check test

help:            ## list targets
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

setup:           ## one-time: Python env, model download, web build
	scripts/setup.sh

index:           ## index a folder of photos: make index DIR=~/Pictures/Trip [INDEX=search/data/index]
	@test -n "$(DIR)" || (echo "usage: make index DIR=path/to/photos"; exit 1)
	D="$$(cd $(DIR) && pwd)" && cd search && PHOTOSEARCH_INDEX=$(abspath $(or $(INDEX),search/data/index)) .venv/bin/python index.py "$$D"

build:           ## build the web app into web/dist
	cd web && npm run build

dev:             ## web dev server with hot reload on :5173 (needs `make start` running)
	cd web && npm run dev

start:           ## run the app on http://127.0.0.1:8000 (the 73k index if built)
	scripts/start.sh

stage:           ## run a staging copy on http://127.0.0.1:8100 (default: the 21k index search/data/index)
	scripts/start.sh --stage $(INDEX)

service:         ## keep the app running at login (launchd, per-user)
	scripts/install-service.sh

unservice:       ## remove the login service
	scripts/install-service.sh --remove

check:           ## tests + web typecheck/build (add ARGS=--ios for the iOS build)
	scripts/check.sh $(ARGS)

test:            ## Python tests only (fast, model-free)
	cd search && .venv/bin/python -m pytest tests -q
