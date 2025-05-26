# ──────────────────────────────
# credit-risk – Makefile raíz
# ──────────────────────────────
.PHONY: up down pipeline pipeline-fast run-pipeline clean-metrics

COMPOSE  = docker compose -f docker/docker-compose.yml
APP_EXEC = $(COMPOSE) exec credit-risk-app bash -c

# 1. Levantar / reconstruir
up:
	$(COMPOSE) up --build -d

# 2. Bajar (limpia huérfanos)
down:
	$(COMPOSE) down --remove-orphans

# 3. SOLO LOS SCRIPTS  ▸  se llama dentro del contenedor
run-pipeline:
	@echo "▶ running pipeline inside container…"
	@export PYTHONPATH=/app ; \
	python src/agents/fetch.py && \
	python src/agents/prep.py && \
	python src/agents/split.py && \
	python src/agents/train_sup.py && \
	python src/agents/train_unsup.py && \
	python src/agents/evaluate.py && \
	python src/agents/register.py

# 4. Pipeline completo (host → contenedor)
pipeline:
	$(APP_EXEC) "make run-pipeline"

# 5. Pipeline rápido (muestra estratificada)
pipeline-fast:
	$(APP_EXEC) "FAST_MODE=true make run-pipeline"

# 6. Limpiar métricas locales
clean-metrics:
	rm -rf metrics && mkdir -p metrics
