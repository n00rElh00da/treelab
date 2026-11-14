from pathlib import Path
from sqlalchemy import select
from app.db import SessionLocal
from app.models import Dataset, DatasetVersion
from app.services.profiling import parse_csv


FIXTURE = Path(__file__).resolve().parent.parent / "sample_data" / "customer_churn_demo.csv"


def seed() -> None:
    content = FIXTURE.read_bytes()
    profile = parse_csv(content, FIXTURE.name)
    with SessionLocal() as db:
        existing = db.scalar(select(Dataset).where(Dataset.slug == "customer-churn-demo", Dataset.is_demo.is_(True)))
        if existing:
            return
        dataset = Dataset(owner_id=None, slug="customer-churn-demo", name="Customer Churn Dataset", description="A small public fixture for onboarding and development testing.", task_type="classification", target_column=profile["target"], is_demo=True)
        db.add(dataset)
        db.flush()
        version = DatasetVersion(dataset_id=dataset.id, version_number=1, original_filename=FIXTURE.name, content_hash=profile["content_hash"], size_bytes=profile["size_bytes"], row_count=profile["row_count"], column_count=profile["column_count"], schema_json=profile["schema"], preview_json=profile["preview"], raw_csv=content, created_by=None)
        db.add(version)
        db.flush()
        dataset.active_version_id = version.id
        db.commit()


if __name__ == "__main__":
    seed()
