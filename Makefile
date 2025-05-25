# ─────────────── Makefile ───────────────
.PHONY: up pipeline

up:
        docker compose -f docker/docker-compose.yml up --build -d

pipeline:
        docker compose -f docker/docker-compose.yml exec credit-risk-app \
                bash -c "python -m src.agents.fetch && \
                        python -m src.agents.prep && \
                        python -m src.agents.split && \
                        python -m src.agents.train_sup && \
                        python -m src.agents.train_unsup && \
                        python -m src.agents.evaluate && \
                        python -m src.agents.register"
