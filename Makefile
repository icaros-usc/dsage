help: ## Print this message.
	@echo "\033[0;1mCommands\033[0m"
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[34;1m%-30s\033[0m %s\n", $$1, $$2}'
.PHONY: help

container.sif: container.def requirements.txt dask_config.yml ## The Singularity container. Requires sudo to run.
	singularity build $@ $<

shell: ## Start a shell in the container.
	singularity shell --cleanenv --nv container.sif
shell-bind: ## Start a shell with GUI and with ./results bound to /results.
	singularity shell --cleanenv --nv --bind ./results:/results container.sif
.PHONY: shell shell-gui

SCHEDULER_FILE = .scheduler_info.json
start-scheduler: ## Starts the Dask scheduler.
	dask-scheduler \
		--scheduler-file $(SCHEDULER_FILE)
start-workers: ## Starts Dask workers. Usage: `make start-workers n=NUM_WORKERS`
	dask-worker \
		--scheduler-file $(SCHEDULER_FILE) \
		--nprocs $(n) \
		--nthreads 1
.PHONY: start-scheduler start-workers

run-local: ## Run locally with 4 workers (see scripts/run_local.sh)
	bash scripts/run_local.sh 4
.PHONY: run-local

test: ## Run unit tests.
	pytest src/
.PHONY: test
