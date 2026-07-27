.PHONY: run network postgres grafana deps up app

run:
	uv run python assistant.py

network:
	@docker network create monitoring 2>/dev/null || true

postgres: network
	@if [ "$$(docker inspect -f '{{.State.Running}}' course-assistant-pg 2>/dev/null)" = "true" ]; then \
		echo "postgres already running"; \
	elif docker ps -a --format '{{.Names}}' | grep -q '^course-assistant-pg$$'; then \
		docker start course-assistant-pg; \
	else \
		docker run -d \
			--name course-assistant-pg \
			--network monitoring \
			-e POSTGRES_USER=user \
			-e POSTGRES_PASSWORD=password \
			-e POSTGRES_DB=course_assistant \
			-p 5432:5432 \
			-v pgdata:/var/lib/postgresql/data \
			postgres:17; \
	fi
	@until docker exec course-assistant-pg pg_isready -U user -d course_assistant >/dev/null 2>&1; do sleep 1; done

grafana: network
	@if [ "$$(docker inspect -f '{{.State.Running}}' grafana 2>/dev/null)" = "true" ]; then \
		echo "grafana already running"; \
	elif docker ps -a --format '{{.Names}}' | grep -q '^grafana$$'; then \
		docker start grafana; \
	else \
		docker run -d \
			--name grafana \
			--network monitoring \
			-p 3000:3000 \
			-v grafana_data:/var/lib/grafana \
			-e GF_SECURITY_ADMIN_PASSWORD=admin \
			grafana/grafana; \
	fi

deps: postgres grafana

up: deps
	docker compose up --build

app:
	streamlit run streamlit_app.py
