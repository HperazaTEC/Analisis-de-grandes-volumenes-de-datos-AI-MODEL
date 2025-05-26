.PHONY: up pipeline

up:
	docker compose -f docker/docker-compose.yml up --build -d

pipeline:
	docker compose -f docker/docker-compose.yml exec credit-risk-app \
		bash -c "export PYTHONPATH=/app && \
			python src/agents/fetch.py && \
			python src/agents/prep.py  && \
			python src/agents/train_sup.py  && \
			python src/agents/train_unsup.py && \
			python src/agents/evaluate.py   && \
			python src/agents/register.py"
