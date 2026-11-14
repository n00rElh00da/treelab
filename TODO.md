# TreeLab — Functional Release Outcomes

## 1. Preserve the product surface while removing mock-only framing
- [x] Keep the supplied TreeLab structure and visual language: landing page, sidebar navigation, dashboard, dataset library, dataset workspace, explorer, preprocessing studio, model laboratory, AutoML laboratory, comparison, explainability, and history.
- [x] Remove the active Iris Dataset, Breast Cancer Wisconsin Dataset, Wine Dataset, and their illustrative experiment records from the application seed/data path.
- [x] Keep only Customer Churn as the initial built-in dataset and keep a separate small demo fixture for testing/demo use without mixing it into user workspaces.
- [x] Replace “simulation mode”, mock metrics, seeded fake progress, and prototype-only upload messaging with database-backed status, errors, and real-run messaging.

## 2. Authentication and private workspaces
- [x] Provide registration, login, logout, current-user state, password hashing, signed HTTP-only sessions, and protected application routes.
- [x] Support `admin` and `user` roles, with authorization designed for future collaboration.
- [x] Ensure each user can access only their own datasets, versions, experiments, model artifacts, and preprocessing configurations; return authorization-safe not-found responses for other users’ IDs.
- [x] Keep the Preview-compatible session cookie behavior and document required environment variables.

## 3. PostgreSQL schema and migrations
- [x] Configure PostgreSQL exclusively through environment variables, including `DATABASE_URL`, with no credentials committed to source.
- [x] Create Alembic migrations from scratch for users, sessions, datasets, versions, preprocessing configurations, experiments, and model artifacts.
- [x] Preserve migration history and include a cleanup migration that removes only known legacy illustrative/mock identifiers, never real user data.
- [x] Provide local PostgreSQL Docker Compose configuration and a small demo seed path that is explicit and separate from production/user data.

## 4. Dataset ingestion and lifecycle
- [x] Uploading a CSV performs real parsing, encoding handling, validation, schema/type inference, target-candidate detection, row/column profiling, preview generation, and persistent version creation.
- [ ] Reject empty/malformed/unsupported or unsafe CSVs with actionable errors and enforce configurable size/row/column limits.
- [x] Allow users to list/search/open/rename/edit metadata/delete datasets and upload/re-upload new versions.
- [x] Preserve immutable dataset versions and prevent deletion of real user data through mock cleanup.
- [x] Show the active Customer Churn dataset and uploaded datasets using the existing library/workspace structure.

## 5. Real preprocessing and model training
- [ ] Persist preprocessing choices per dataset version, including target, selected features, missing-value handling, categorical encoding, scaling, duplicate handling, imbalance strategy, and train/test split.
- [x] Train real Decision Tree and Random Forest models through scikit-learn pipelines.
- [x] Limit Decision Tree criterion options to Gini and Entropy only.
- [x] Persist experiment status, parameters, metrics, confusion matrix, duration, model metadata, and serialized model artifact under the authenticated owner.
- [x] Provide real accuracy, precision, recall, F1, applicable ROC-AUC/classification report, and failure handling.

## 6. Real explainability and review
- [ ] Compute and persist model feature importance, permutation importance, prediction probabilities, sample predictions, and a tree/estimator representation when supported.
- [x] Populate the existing explainability, comparison, and experiment history surfaces from persisted results.
- [x] Ensure comparison can only compare experiments owned by the current user and supports empty/insufficient-selection states.

## 7. Real FLAML AutoML
- [x] Implement real FLAML execution and persist the winning estimator, configuration, metrics, explainability boundary, and duration as an experiment.
- [x] Controlled mode uses a time budget in seconds and rejects values above 90 seconds.
- [x] Extended mode searches a backend-defined broad estimator space for the best result and does not require a user-selected fixed algorithm list.
- [x] Show running, completed, and failed states without fabricated metrics.

## 8. Preview, Docker, and Azure-ready delivery
- [x] Serve `/api/health` and the complete static route manifest at `/manus-routes.json`.
- [x] Keep Preview listening on port 3000 and serve the application from a production-capable FastAPI container.
- [x] Provide Dockerfile, Compose configuration, `.env.example`, dependency pinning, migration commands, and Azure deployment documentation for Azure for Students.
- [x] Do not deploy, provision resources, or incur cloud costs without explicit approval.

## 9. Validation and handoff
- [ ] Run syntax/type/static checks, migration checks where PostgreSQL is available, API-level tests for auth/ownership/profiling/training validation, and a production build/container configuration check.
- [ ] Resolve actionable diagnostics before checkpointing.
- [ ] Commit the completed implementation to the managed canonical remote without enabling auto-publish.
