.PHONY: build deploy local test setup-ssm

# ── Build & Deploy ────────────────────────────────────────────────────────────
build:
	sam build

deploy: build
	sam deploy

# ── Local testing (requires Docker) ──────────────────────────────────────────
# Create env.json from .env.example values for sam local
local: build
	sam local start-api --env-vars env.json --warm-containers EAGER

# ── Smoke test ────────────────────────────────────────────────────────────────
test:
	API_URL=$${API_URL:-http://localhost:3000} python scripts/test_api.py

# ── Bootstrap SSM secrets (run once before first deploy) ─────────────────────
setup-ssm:
	@test -n "$(OPENAI_API_KEY)"    || (echo "ERROR: OPENAI_API_KEY is not set"    && exit 1)
	@test -n "$(LANGCHAIN_API_KEY)" || (echo "ERROR: LANGCHAIN_API_KEY is not set" && exit 1)
	aws ssm put-parameter \
		--name  /langsmith-rag/openai-api-key \
		--value "$(OPENAI_API_KEY)" \
		--type  SecureString \
		--overwrite
	aws ssm put-parameter \
		--name  /langsmith-rag/langsmith-api-key \
		--value "$(LANGCHAIN_API_KEY)" \
		--type  SecureString \
		--overwrite
	@echo "SSM parameters created."
