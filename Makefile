up:
	docker compose -f docker/docker-compose.yml up --build -d

pipeline:
	docker compose -f docker/docker-compose.yml exec credit-risk-app dvc pull
	docker compose -f docker/docker-compose.yml exec credit-risk-app dvc repro
