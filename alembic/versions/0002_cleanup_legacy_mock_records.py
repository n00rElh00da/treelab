"""remove legacy illustrative dataset records only

Revision ID: 0002_cleanup_legacy_mock_records
Revises: 0001_initial
"""
from alembic import op
from sqlalchemy import text

revision = "0002_cleanup_legacy_mock_records"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade():
    # Only demo rows with known legacy slugs are touched. Real user-owned rows are not targeted.
    op.execute(text("DELETE FROM datasets WHERE is_demo = TRUE AND slug IN ('iris', 'breast-cancer', 'wine', 'customer-churn-mock')"))


def downgrade():
    # Deliberately irreversible: the removed illustrative rows were never real user data.
    pass
