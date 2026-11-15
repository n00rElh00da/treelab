import os
from pathlib import Path
from uuid import uuid4

import pytest

playwright = pytest.importorskip("playwright.sync_api")
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import expect, sync_playwright


BASE_URL = os.getenv("TREELAB_E2E_BASE_URL", "http://localhost:3000")


@pytest.fixture
def page():
    with sync_playwright() as playwright_context:
        browser = playwright_context.chromium.launch()
        page = browser.new_page()
        try:
            page.goto(BASE_URL, wait_until="networkidle")
        except PlaywrightError as exc:
            browser.close()
            pytest.skip(f"TreeLab is not running at {BASE_URL}: {exc}")
        yield page
        browser.close()


def test_auth_modal_navigation_and_controls(page) -> None:
    page.get_by_role("button", name="Log in").click()
    expect(page.get_by_role("heading", name="Log in to TreeLab")).to_be_visible()

    page.get_by_role("button", name="Need an account").click()
    expect(page.get_by_role("heading", name="Create your TreeLab account")).to_be_visible()

    page.get_by_role("button", name="Already registered?").click()
    expect(page.get_by_role("heading", name="Log in to TreeLab")).to_be_visible()

    page.locator("#modal-root .modal-header .icon-btn").click()
    expect(page.locator("#modal-root .modal")).to_have_count(0)


@pytest.mark.e2e
def test_dataset_crud_on_explicitly_isolated_environment(page, tmp_path: Path) -> None:
    if os.getenv("TREELAB_E2E_ALLOW_MUTATION") != "1" or not os.getenv("TREELAB_E2E_DATABASE_URL"):
        pytest.skip("Requires an explicitly isolated test database and TREELAB_E2E_ALLOW_MUTATION=1")

    email = f"e2e-{uuid4().hex}@example.test"
    page.get_by_role("button", name="Create account").first.click()
    page.get_by_label("Email").fill(email)
    page.get_by_label("Name").fill("E2E User")
    page.get_by_label("Password").fill("correct-horse-battery")
    page.get_by_role("button", name="Create account").last.click()
    expect(page.get_by_text("Dashboard")).to_be_visible()

    page.get_by_role("button", name="Add dataset").click()
    page.get_by_label("Dataset name").fill("E2E Dataset")
    page.locator('input[type="file"]').set_input_files(
        {
            "name": "e2e.csv",
            "mimeType": "text/csv",
            "buffer": b"target,feature\nno,1\nyes,2\nno,3\nyes,4\n",
        }
    )
    page.get_by_role("button", name="Upload and profile").click()
    expect(page.get_by_text("E2E Dataset")).to_be_visible()

    page.get_by_role("button", name="My Datasets").click()
    page.get_by_role("button", name="Edit dataset").click()
    page.get_by_label("Name").fill("E2E Dataset Renamed")
    page.get_by_role("button", name="Save changes").click()
    expect(page.get_by_text("E2E Dataset Renamed")).to_be_visible()

    page.get_by_role("button", name="Delete dataset").click()
    page.get_by_role("button", name="Delete dataset").last.click()
    expect(page.get_by_text("E2E Dataset Renamed")).to_have_count(0)
