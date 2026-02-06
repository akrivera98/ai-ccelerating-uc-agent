IMAGE ?= uc_eval:local

build:
	docker build --platform=linux/amd64 -t $(IMAGE) .

run:
	docker run --platform=linux/amd64 --rm -it \
		-v "$(CURDIR)/.uv_cache":/root/.cache/uv \
		-v "$(CURDIR)/.uv_data":/root/.local/share/uv \
		-v "$(CURDIR)/submission/":/app/ingested_program/ \
		-v "$(CURDIR)/data/Test_Data/":/app/input_data/ \
		-v "$(CURDIR)/submission/":/app/output/ \
		-v "$(CURDIR)/submission/":/app/input/res/ \
		-v "$(CURDIR)/data/Test_Data/":/app/input/ref/ \
		-v "$(CURDIR)/eval/":/app/scripts/ \
		-w /app \
		-e PYTHON_JULIACALL_EXE=/usr/local/bin/julia \
		-e PYTHON_JULIACALL_PROJECT=/root/.julia/environments/v1.11 \
		$(IMAGE) \
		bash -lc "uv sync --project /app/ingested_program && \
				  uv run  --project /app/ingested_program python /app/scripts/ingestion.py && \
				  uv run  --project /app/ingested_program python /app/scripts/scoring.py"