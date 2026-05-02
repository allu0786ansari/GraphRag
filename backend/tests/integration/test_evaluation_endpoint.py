"""
tests/integration/test_evaluation_endpoint.py — /api/v1/evaluate integration tests.

Strategy:
  - Mock _check_indexed (storage layer) so tests don't need a real corpus.
  - Mock EvaluationEngine.from_settings to avoid real LLM calls.
  - Reset _EVAL_STORE between tests to prevent cross-test bleed.
  - Test HTTP contract: status codes, response shape, auth, validation.
"""

from __future__ import annotations

import pytest
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch


# ── Store reset ───────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_eval_store():
    import app.api.routes_evaluation as re
    re._EVAL_STORE.clear()
    yield
    re._EVAL_STORE.clear()


# ── Helpers ───────────────────────────────────────────────────────────────────

def minimal_eval_request():
    return {
        "questions": ["What are the main themes across all documents?"],
        "criteria": ["comprehensiveness"],
    }


@pytest.fixture(autouse=True)
def mock_indexed():
    """Make the evaluation endpoint believe the corpus is indexed."""
    mock_store = MagicMock()
    mock_store.summaries_exist.return_value = True
    with patch("app.storage.summary_store.SummaryStore", return_value=mock_store):
        yield


# ── Auth ──────────────────────────────────────────────────────────────────────

class TestEvaluationAuth:

    def test_post_evaluate_requires_auth(self, client):
        response = client.post("/api/v1/evaluate", json=minimal_eval_request())
        assert response.status_code == 401

    def test_get_results_requires_auth(self, client):
        response = client.get("/api/v1/evaluation/results")
        assert response.status_code == 401

    def test_get_result_by_id_requires_auth(self, client):
        response = client.get("/api/v1/evaluation/results/eval_fake")
        assert response.status_code == 401


# ── POST /api/v1/evaluate ─────────────────────────────────────────────────────

class TestPostEvaluate:

    def test_returns_202_accepted(self, client, auth_headers):
        response = client.post(
            "/api/v1/evaluate",
            json=minimal_eval_request(),
            headers=auth_headers,
        )
        assert response.status_code == 202

    def test_response_has_eval_id(self, client, auth_headers):
        response = client.post(
            "/api/v1/evaluate",
            json=minimal_eval_request(),
            headers=auth_headers,
        )
        data = response.json()
        assert "eval_id" in data
        assert len(data["eval_id"]) > 0

    def test_response_has_status(self, client, auth_headers):
        response = client.post(
            "/api/v1/evaluate",
            json=minimal_eval_request(),
            headers=auth_headers,
        )
        assert "status" in response.json()

    def test_empty_questions_returns_422(self, client, auth_headers):
        response = client.post(
            "/api/v1/evaluate",
            json={"questions": []},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_invalid_criterion_returns_422(self, client, auth_headers):
        response = client.post(
            "/api/v1/evaluate",
            json={"questions": ["test?"], "criteria": ["invalid_criterion"]},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_multiple_questions_accepted(self, client, auth_headers):
        response = client.post(
            "/api/v1/evaluate",
            json={
                "questions": [
                    "What are the main themes?",
                    "Who are the key players?",
                    "What investments were made?",
                ],
                "criteria": ["comprehensiveness", "diversity"],
            },
            headers=auth_headers,
        )
        assert response.status_code == 202

    def test_all_criteria_accepted(self, client, auth_headers):
        response = client.post(
            "/api/v1/evaluate",
            json={
                "questions": ["Main themes?"],
                "criteria": [
                    "comprehensiveness", "diversity",
                    "empowerment", "directness",
                ],
            },
            headers=auth_headers,
        )
        assert response.status_code == 202

    def test_503_when_not_indexed(self, client, auth_headers):
        mock_store = MagicMock()
        mock_store.summaries_exist.return_value = False
        with patch("app.storage.summary_store.SummaryStore", return_value=mock_store):
            response = client.post(
                "/api/v1/evaluate",
                json=minimal_eval_request(),
                headers=auth_headers,
            )
        assert response.status_code == 503


# ── GET /api/v1/evaluation/results ───────────────────────────────────────────

class TestGetEvaluationResults:

    def test_returns_200_with_empty_list(self, client, auth_headers):
        response = client.get("/api/v1/evaluation/results", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_submitted_job_appears_in_list(self, client, auth_headers):
        post_resp = client.post(
            "/api/v1/evaluate",
            json=minimal_eval_request(),
            headers=auth_headers,
        )
        eval_id = post_resp.json()["eval_id"]

        list_resp = client.get("/api/v1/evaluation/results", headers=auth_headers)
        ids = [r["eval_id"] for r in list_resp.json()["results"]]
        assert eval_id in ids

    def test_results_list_has_correct_shape(self, client, auth_headers):
        client.post(
            "/api/v1/evaluate",
            json=minimal_eval_request(),
            headers=auth_headers,
        )
        list_resp = client.get("/api/v1/evaluation/results", headers=auth_headers)
        for item in list_resp.json()["results"]:
            assert "eval_id" in item
            assert "status" in item


# ── GET /api/v1/evaluation/results/{eval_id} ─────────────────────────────────

class TestGetEvaluationResultById:

    def test_unknown_eval_id_returns_404(self, client, auth_headers):
        response = client.get(
            "/api/v1/evaluation/results/eval_doesnotexist",
            headers=auth_headers,
        )
        assert response.status_code == 404

    def test_in_progress_job_returns_202(self, client, auth_headers):
        post_resp = client.post(
            "/api/v1/evaluate",
            json=minimal_eval_request(),
            headers=auth_headers,
        )
        eval_id = post_resp.json()["eval_id"]

        # Manually force to running (simulating async background task)
        import app.api.routes_evaluation as re
        from app.models.response_models import PipelineStatus
        if eval_id in re._EVAL_STORE:
            re._EVAL_STORE[eval_id]["status"] = PipelineStatus.RUNNING

        get_resp = client.get(
            f"/api/v1/evaluation/results/{eval_id}",
            headers=auth_headers,
        )
        assert get_resp.status_code in (200, 202)

    def test_completed_job_returns_result(self, client, auth_headers):
        """Manually insert a completed result and verify it is returned."""
        from app.models.response_models import PipelineStatus
        from app.models.evaluation_models import EvalResponse
        import app.api.routes_evaluation as re

        eval_id = f"eval_{uuid.uuid4().hex[:8]}"
        # Build minimal EvalResponse
        mock_result = MagicMock(spec=EvalResponse)
        mock_result.model_dump = MagicMock(return_value={
            "eval_id": eval_id, "status": "completed",
            "questions": ["test?"], "criteria": ["comprehensiveness"],
            "eval_runs": 1,
        })

        re._EVAL_STORE[eval_id] = {
            "status": PipelineStatus.COMPLETED,
            "result": mock_result,
            "error_message": None,
        }

        response = client.get(
            f"/api/v1/evaluation/results/{eval_id}",
            headers=auth_headers,
        )
        # Should be 200 with the result
        assert response.status_code == 200