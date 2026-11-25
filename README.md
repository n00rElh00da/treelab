# TreeLab — private ML experimentation workspace

TreeLab keeps the supplied prototype's dark research-console interface but now connects it to a real Python backend. Users can register and log in, upload CSV files, inspect persisted versions, configure preprocessing, train scikit-learn Decision Trees/Random Forests, run FLAML AutoML, compare experiments, and review real explainability artifacts.

## What changed from the prototype

The active application no longer seeds Iris, Breast Cancer, Wine, or mock experiment records. The only built-in catalog entry is a small public Customer Churn fixture used for onboarding and development. It is separate from real user data. Uploaded datasets are private to the authenticated owner, versioned, persisted, and soft-deleted so accidental deletion does not erase real records.

Decision Tree criterion is intentionally limited to `gini` and `entropy`. Controlled AutoML accepts a time budget in seconds from 1 through 90. Extended AutoML delegates estimator selection to a broader backend-defined FLAML search space rather than a fixed user checklist.

## Local development

The recommended path is Docker Compose, which runs PostgreSQL and the FastAPI app locally without provisioning any cloud resources:

```bash
cp .env.example .env
# Set unique POSTGRES_PASSWORD and SESSION_SECRET values before sharing the environment.
docker compose up --build
```

Open `http://localhost:3000`. The first registered account becomes `admin`; subsequent accounts are `user`. The Compose command applies Alembic migrations and seeds the public Customer Churn fixture. To run the app directly on a host Python installation, start PostgreSQL separately, install the dependencies, export `DATABASE_URL`, then run:

```bash
python3 -m pip install -r requirements.txt
alembic upgrade head
python -m app.seed_demo
uvicorn app.main:app --host 0.0.0.0 --port 3000
```

The application reads `DATABASE_URL`, `SESSION_SECRET`, `APP_ENV`, `COOKIE_SECURE`, `COOKIE_SAMESITE`, `PORT`, upload limits, and `EXPLAINABILITY_MAX_ROWS` from the environment. `EXPLAINABILITY_MAX_ROWS` bounds the holdout rows used for permutation importance (default 1,000); it does not change model training or persisted prediction samples. `DATABASE_URL` is required for authenticated persistence; there is no committed database credential fallback. Passwords use Argon2id when the pinned `argon2-cffi` dependency is installed, while the verifier retains compatibility with the prior scrypt format for constrained migrations. Browser mutation requests with an Origin or Referer are checked against the forwarded request origin or configured CORS origins. Never commit `.env` or credentials. Compose keeps PostgreSQL on the private container network rather than publishing port 5432 to the host. For Preview and production, set `APP_ENV=production` and `COOKIE_SECURE=true`, and use protected environment/secret configuration for `DATABASE_URL`, `SESSION_SECRET`, and `POSTGRES_PASSWORD`; the application rejects production settings that leave secure cookies disabled. Without a DSN, `/api/health` reports an explicit degraded/unconfigured state and the app does not silently fall back to mock data.

## Database and migrations

The PostgreSQL schema is defined in `app/models.py` and created through checked-in Alembic migrations under `alembic/versions/`. Run `alembic upgrade head` for a new environment and `alembic downgrade -1` only when intentionally rolling back. Migration `0002_cleanup_legacy_mock_records` targets only known demo slugs and `is_demo = true`; it does not target user-owned data. The initial Customer Churn fixture is inserted only by the explicit `python -m app.seed_demo` command.

The initial schema includes users, sessions, datasets, immutable dataset versions, preprocessing configurations, experiments, and serialized model artifacts. Every user-owned query is scoped by the authenticated owner ID. Demo datasets are read-only and public to authenticated users.

## API overview

`/api/auth/register`, `/api/auth/login`, `/api/auth/logout`, and `/api/auth/me` provide application authentication. `/api/datasets` provides private dataset listing, CSV creation, metadata edits, soft deletion, version uploads, and previews. `/api/experiments` provides owner-scoped history, real model training, owner-scoped serialized model downloads, and real FLAML AutoML. Invalid preprocessing options, target/feature leakage, and datasets without a usable target are rejected with actionable errors. `/api/health` reports application/database readiness. `/manus-routes.json` is served as the required page manifest.


## Validation

The backend can be syntax-checked with `python3 -m compileall -q app alembic`, imported with `python3 -c 'from app.main import app'`, and checked with `node --check script.js`. With PostgreSQL available, run the migrations, seed fixture, start the app, and verify `GET /api/health`, `GET /manus-routes.json`, registration/login, CSV upload, training, and AutoML through the UI/API. The sandbox used for this implementation does not provide a running local PostgreSQL daemon or Docker, so database execution remains documented and ready for the user’s local environment.

Development tests are declared in `requirements-dev.txt`:

```bash
python3 -m pytest -q
python3 -m playwright install chromium
TREELAB_E2E_BASE_URL=http://localhost:3000 python3 -m pytest tests/e2e -q
```

The browser Dataset CRUD test is skipped unless `TREELAB_E2E_ALLOW_MUTATION=1` and `TREELAB_E2E_DATABASE_URL` identify an explicitly isolated test database. It must not be run against a normal user database.
