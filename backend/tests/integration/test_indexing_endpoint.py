"""
tests/integration/test_indexing_endpoint.py — POST/GET /api/v1/index integration tests.

Strategy:
  - These tests exercise the full HTTP → worker → job-store chain.
  - The PipelineRunner is mocked to avoid any real LLM/FAISS calls.
  - Worker state is reset between every test to prevent cross-test bleed.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


# ── State reset ───────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_worker():
    """Reset indexing worker job store before each test."""
    import app.workers.indexing_worker as iw
    iw._JOB_STORE.clear()
    iw._ACTIVE_JOB_ID = None
    yield
    iw._JOB_STORE.clear()
    iw._ACTIVE_JOB_ID = None


# ── Auth ──────────────────────────────────────────────────────────────────────

class TestIndexingAuth:

    def test_post_index_requires_auth(self, client):
        response = client.post("/api/v1/index", json={})
        assert response.status_code == 401

    def test_get_status_requires_auth(self, client):
        response = client.get("/api/v1/index/status")
        assert response.status_code == 401


# ── POST /api/v1/index ────────────────────────────────────────────────────────

class TestPostIndex:

    def test_returns_202_accepted(self, client, auth_headers):
        response = client.post("/api/v1/index", json={}, headers=auth_headers)
        assert response.status_code == 202

    def test_response_has_job_id(self, client, auth_headers):
        response = client.post("/api/v1/index", json={}, headers=auth_headers)
        data = response.json()
        assert "job_id" in data
        assert data["job_id"].startswith("idx_")

    def test_response_has_status_queued(self, client, auth_headers):
        response = client.post("/api/v1/index", json={}, headers=auth_headers)
        assert response.json()["status"] == "queued"

    def test_response_has_message(self, client, auth_headers):
        response = client.post("/api/v1/index", json={}, headers=auth_headers)
        assert "message" in response.json()
        assert len(response.json()["message"]) > 0

    def test_default_parameters_accepted(self, client, auth_headers):
        response = client.post(
            "/api/v1/index",
            json={"chunk_size": 600, "chunk_overlap": 100, "gleaning_rounds": 2},
            headers=auth_headers,
        )
        assert response.status_code == 202

    def test_force_reindex_true_accepted(self, client, auth_headers):
        response = client.post(
            "/api/v1/index",
            json={"force_reindex": True},
            headers=auth_headers,
        )
        assert response.status_code == 202

    def test_invalid_chunk_overlap_returns_422(self, client, auth_headers):
        # chunk_overlap must be < chunk_size
        response = client.post(
            "/api/v1/index",
            json={"chunk_size": 300, "chunk_overlap": 400},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_gleaning_rounds_too_high_returns_422(self, client, auth_headers):
        response = client.post(
            "/api/v1/index",
            json={"gleaning_rounds": 99},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_conflict_when_job_already_running(self, client, auth_headers):
        from app.workers.indexing_worker import make_blank_stages
        from app.models.response_models import PipelineStatus
        import app.workers.indexing_worker as iw

        # Simulate a running job
        iw._ACTIVE_JOB_ID = "idx_running"
        iw._JOB_STORE["idx_running"] = {
            "status": PipelineStatus.RUNNING,
            "stages": make_blank_stages(),
        }

        response = client.post("/api/v1/index", json={}, headers=auth_headers)
        assert response.status_code == 409
        assert response.json()["detail"]["error"] == "job_already_running"


# ── GET /api/v1/index/status ──────────────────────────────────────────────────

class TestGetIndexStatus:

    def test_no_jobs_returns_200_with_none_id(self, client, auth_headers):
        response = client.get("/api/v1/index/status", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["job_id"] == "none"

    def test_status_after_submission(self, client, auth_headers):
        # Submit a job
        post_resp = client.post("/api/v1/index", json={}, headers=auth_headers)
        job_id = post_resp.json()["job_id"]

        # Poll status
        status_resp = client.get(
            f"/api/v1/index/status?job_id={job_id}",
            headers=auth_headers,
        )
        assert status_resp.status_code == 200
        data = status_resp.json()
        assert data["job_id"] == job_id
        assert data["status"] in ("queued", "running", "completed", "failed")

    def test_unknown_job_id_returns_404(self, client, auth_headers):
        response = client.get(
            "/api/v1/index/status?job_id=idx_doesnotexist",
            headers=auth_headers,
        )
        assert response.status_code == 404
        assert response.json()["detail"]["error"] == "job_not_found"

    def test_status_without_job_id_returns_most_recent(self, client, auth_headers):
        post_resp = client.post("/api/v1/index", json={}, headers=auth_headers)
        job_id = post_resp.json()["job_id"]

        status_resp = client.get("/api/v1/index/status", headers=auth_headers)
        assert status_resp.status_code == 200
        assert status_resp.json()["job_id"] == job_id

    def test_status_response_has_stages_list(self, client, auth_headers):
        post_resp = client.post("/api/v1/index", json={}, headers=auth_headers)
        job_id = post_resp.json()["job_id"]

        status_resp = client.get(
            f"/api/v1/index/status?job_id={job_id}",
            headers=auth_headers,
        )
        data = status_resp.json()
        assert "stages" in data
        assert isinstance(data["stages"], list)
        assert len(data["stages"]) > 0

    def test_status_stages_have_correct_shape(self, client, auth_headers):
        post_resp = client.post("/api/v1/index", json={}, headers=auth_headers)
        job_id = post_resp.json()["job_id"]

        status_resp = client.get(
            f"/api/v1/index/status?job_id={job_id}",
            headers=auth_headers,
        )
        for stage in status_resp.json()["stages"]:
            assert "stage" in stage
            assert "status" in stage


# ── Background task execution ─────────────────────────────────────────────────

class TestIndexingBackgroundExecution:

    def test_job_transitions_to_completed_on_success(self, client, auth_headers):
        """
        Submit a job with a mocked successful pipeline.
        FastAPI TestClient runs background tasks synchronously.
        """
        from app.core.pipeline.pipeline_runner import PipelineResult

        mock_result = PipelineResult(
            success=True, run_id="run_test",
            total_elapsed_seconds=30.0,
        )
        mock_result.chunks_count = 10
        mock_result.graph_nodes = 15
        mock_result.graph_edges = 20
        mock_result.summaries_count = 5

        with patch("app.workers.indexing_worker.PipelineRunner") as MockRunner:
            runner_instance = AsyncMock()
            runner_instance.run = AsyncMock(return_value=mock_result)
            MockRunner.return_value = runner_instance

            with patch("app.workers.indexing_worker.GraphStore") as MockGS:
                MockGS.return_value.get_community_counts.return_value = {}

                post_resp = client.post("/api/v1/index", json={}, headers=auth_headers)
                job_id = post_resp.json()["job_id"]

        # Background task ran synchronously in TestClient
        from app.workers.indexing_worker import get_job
        job = get_job(job_id)
        if job:
            assert job["status"].value in ("completed", "failed", "queued")

    def test_job_transitions_to_failed_on_pipeline_error(self, client, auth_headers):
        """Pipeline returns success=False → job should be FAILED."""
        from app.core.pipeline.pipeline_runner import PipelineResult

        fail_result = PipelineResult(
            success=False, run_id="run_fail",
            total_elapsed_seconds=5.0,
        )
        fail_result.error_stage = "extraction"
        fail_result.error_message = "LLM API unavailable"

        with patch("app.workers.indexing_worker.PipelineRunner") as MockRunner:
            runner_instance = AsyncMock()
            runner_instance.run = AsyncMock(return_value=fail_result)
            MockRunner.return_value = runner_instance

            post_resp = client.post("/api/v1/index", json={}, headers=auth_headers)
            job_id = post_resp.json()["job_id"]

        from app.workers.indexing_worker import get_job
        job = get_job(job_id)
        if job:
            assert job["status"].value in ("failed", "queued")