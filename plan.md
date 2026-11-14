# TreeLab — Functional ML Workspace Plan

## Product scope

TreeLab will keep the supplied prototype's dark research-console visual language and single-page workspace structure, but replace illustrative-only behavior with a real application. The active built-in catalog will contain only the Customer Churn dataset. The former Iris, Breast Cancer, and Wine records and their mock experiments will not be seeded into the active application. A separate, clearly marked demo fixture will exist only for development/testing and will never be mixed into a user's private workspace.

The app will support registration/login, private per-user workspaces, dataset CRUD and versioning, CSV ingestion and validation, real model training, explainability, and real AutoML. PostgreSQL is external to Webdev and is configured through `DATABASE_URL`; local development uses Docker Compose/PostgreSQL, while Azure Database for PostgreSQL Flexible Server is documented as a future deployment target. No cloud resources are provisioned by this implementation.

## Architecture

- **Backend:** Python 3.12, FastAPI, SQLAlchemy 2.x, Alembic, PostgreSQL via psycopg, pandas, scikit-learn, and FLAML.
- **Frontend:** existing HTML/CSS/vanilla JavaScript prototype, adapted to call JSON APIs and render real server data while preserving the existing navigation, workspace tabs, charts, forms, and visual hierarchy.
- **Authentication:** application-owned email/password registration and login, Argon2id password hashing where available with a documented bcrypt fallback, signed HTTP-only session cookie, CSRF-safe same-origin mutation checks, and roles `admin`/`user`. The cookie name is `webdev_app_session`; Preview cookies use `SameSite=None; Secure`, while plain local HTTP uses `SameSite=Lax`.
- **Persistence:** PostgreSQL tables for users, sessions, datasets, dataset versions, experiments, models, and preprocessing configurations. Every user-owned table has an owner/user relationship enforced in query filters and service-layer authorization.
- **Training:** synchronous API jobs for the first functional release, with a job/status abstraction so execution can move to a worker later. Dataset versions are immutable; experiment records capture parameters, preprocessing, metrics, explainability artifacts, and model metadata.
- **Storage:** CSV content is stored in PostgreSQL as compressed text/bytes for the initial local/Preview release, with an explicit repository seam for Azure Blob Storage later. Database rows retain file metadata, schema, preview, and version references.

## Data model

- `users`: id, email, name, password hash, role, active flag, timestamps.
- `sessions`: id, user id, hashed token, expiry, timestamps.
- `datasets`: id, owner id, name, description, task type, target column, active version id, deleted-at, timestamps.
- `dataset_versions`: id, dataset id, version number, original filename, content hash, size, row/column counts, schema JSON, preview JSON, raw CSV bytes/text, created-by, timestamps.
- `preprocessing_configs`: id, dataset version id, owner id, target, selected features, missing-value strategy, categorical encoding, scaling, duplicate handling, class-imbalance strategy, train/test split, created-at.
- `experiments`: id, owner id, dataset/version ids, model type, status, parameters JSON, metrics JSON, confusion matrix JSON, feature importance JSON, permutation importance JSON, prediction samples JSON, duration, error, timestamps.
- `model_artifacts`: id, experiment id, owner id, serialized model bytes, framework/version metadata, created-at.

Soft deletion is used for real user datasets so accidental deletion cannot erase user-owned records. The cleanup migration removes only known illustrative/mock records if they exist by their explicit legacy identifiers; it never touches user-owned records.

## API surface

- `GET /api/health` — unauthenticated readiness response.
- `GET /api/auth/me`, `POST /api/auth/register`, `POST /api/auth/login`, `POST /api/auth/logout`.
- `GET /api/datasets`, `POST /api/datasets`, `GET /api/datasets/{id}`, `PATCH /api/datasets/{id}`, `DELETE /api/datasets/{id}`.
- `POST /api/datasets/{id}/versions` — multipart CSV upload, parse/validate/profile/target detection and version creation.
- `GET /api/datasets/{id}/versions/{version_id}/preview` — preview/schema/profile.
- `GET /api/experiments`, `GET /api/experiments/{id}`.
- `POST /api/experiments/train` — Decision Tree or Random Forest with validated parameters and real metrics/explainability.
- `POST /api/experiments/automl` — FLAML controlled mode (1–90 seconds) or extended mode (search space chosen by the backend).
- `GET /api/experiments/{id}/model` — authorization-protected model metadata/download boundary.

## ML rules

CSV parsing rejects empty files, malformed CSV, unsupported encodings, excessive columns/rows, and datasets without usable target candidates. The target is detected from common names (`target`, `label`, `class`, `churn`, `y`) or defaults to the last column only when it has a bounded number of distinct values; the user can change it before training. Numeric and categorical columns are handled with a scikit-learn `ColumnTransformer`, imputers, one-hot/ordinal encoding, optional scaling, and a deterministic stratified split when possible.

Decision Tree criterion is limited to `gini` and `entropy`; Random Forest uses supported controls. Metrics include accuracy, precision, recall, F1, ROC-AUC when applicable, classification report, and confusion matrix. Explainability includes native tree/forest feature importance, permutation importance on the holdout set, and sample prediction probabilities. Models are serialized only after successful runs.

FLAML controlled mode accepts `time_budget` in seconds and rejects values above 90. Extended mode uses a backend-defined estimator space and does not expose a fixed estimator checklist. AutoML outputs are persisted as experiments and use the same ownership rules as manual training.

## Frontend integration

The existing landing page, dashboard, dataset library, workspace tabs, charts, preprocessing controls, model lab, AutoML lab, comparison, explainability, and history remain the presentation structure. Mock constants and simulated timers are replaced with API loading states, empty/error states, real upload progress, real result rendering, and auth-aware navigation. Dataset menu actions become rename/version/delete actions. The UI makes the active user, private workspace, database-backed status, and real-run status visible; it no longer calls results simulated.

## Project structure

```text
app/
  main.py                 # FastAPI app, static serving, health and lifecycle
  config.py               # environment-driven settings
  db.py                   # SQLAlchemy engine/session and transaction helpers
  models.py               # ORM models
  schemas.py              # request/response validation
  auth.py                 # hashing, sessions, user dependency, roles
  api/
    auth.py               # registration/login/logout/me
    datasets.py           # CRUD, upload, versions, profiling
    experiments.py        # train, AutoML, comparison/history
  services/
    profiling.py          # CSV parsing, validation, schema/target inference
    preprocessing.py      # sklearn pipeline construction
    training.py           # real estimators, metrics, explainability
    automl.py             # FLAML controlled/extended execution
    storage.py            # database-backed initial storage seam
alembic/
  versions/               # checked-in PostgreSQL migrations
static/
  index.html
  styles.css
  script.js
  manus-routes.json
sample_data/customer_churn_demo.csv
Dockerfile
docker-compose.yml
requirements.txt
.env.example
README.md
plan.md
TODO.md
```

## Deployment and build workflow

Local development runs PostgreSQL and the FastAPI app with Docker Compose or a host Python environment using `DATABASE_URL`. Preview runs the same app on port 3000 with an external `DATABASE_URL` secret when supplied; without a database it remains visibly unavailable rather than silently reverting to mock data. Production uses the included Dockerfile, runtime `PORT`, `DATABASE_URL`, `SESSION_SECRET`, and optional upload limits. The README documents Azure Container Apps/App Service plus Azure Database for PostgreSQL Flexible Server, migrations, secrets, TLS, and the explicit approval gate before provisioning. No publication or cloud provisioning is enabled by this task.
