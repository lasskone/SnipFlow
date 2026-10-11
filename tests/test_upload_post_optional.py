"""Upload-Post key is optional for generation/preview/download, required to publish.

The split: a self-hosted user only needs a Gemini key (or a local LLM) to
generate clips, preview them and download them. The Upload-Post key is the
social-publishing credential, so ``/api/social/post`` alone demands it.

The important half of that contract is the failure mode. A publish request that
arrives with no key must come back as a *validation error*. A broken
intermediate revision returned 200 with clip info and a ``generated_only``
status instead — which every caller reads as success: ``ResultCard.jsx`` would
toast "Posted successfully!", ``ScheduleWeekModal`` would count the clip as
scheduled, and the MCP server / CLI / n8n skill would all report the post as
live when nothing was ever uploaded. A caller that asked to publish must be told
it could not.
"""
import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app import app, jobs

client = TestClient(app)


class TestUploadPostOptional:
    """Upload-Post is optional for core use, mandatory for publishing."""

    @pytest.fixture(autouse=True)
    def setup_jobs(self):
        # Clear jobs before each test
        jobs.clear()
        yield
        jobs.clear()

    def test_publish_without_upload_post_key_is_a_validation_error(self, tmp_path):
        """Posting with no Upload-Post key must 400, not 200-with-clip-info."""
        job_id = "test-job-123"
        jobs[job_id] = {
            "result": {
                "clips": [
                    {
                        "video_url": "/videos/test-job-123/clip1.mp4",
                        "title": "Test Clip",
                        "video_description_for_tiktok": "Test description"
                    }
                ]
            }
        }

        # Create mock video file so the request gets past the file check and
        # would otherwise reach the vendor call.
        job_dir = tmp_path / job_id
        job_dir.mkdir()
        (job_dir / "clip1.mp4").write_bytes(b"mock video content")

        with pytest.MonkeyPatch().context() as mp:
            mp.delenv("UPLOAD_POST_API_KEY", raising=False)
            mp.setattr("app.OUTPUT_DIR", str(tmp_path))
            # If the missing key were not caught up front, the handler would try
            # to POST to the real vendor; make that obvious if it ever happens.
            mp.setattr("app._post_video_blocking",
                       lambda *a, **k: pytest.fail("vendor was called with no key"))

            response = client.post("/api/social/post", json={
                "job_id": job_id,
                "clip_index": 0,
                "platforms": ["tiktok"]
            })

        assert response.status_code == 400, response.text
        detail = response.json()["detail"].lower()
        assert "upload-post" in detail
        assert "key" in detail
        # The old silent-success shape must be gone.
        assert response.json().get("status") != "generated_only"

    def test_generation_preview_download_need_no_upload_post_key(self, tmp_path):
        """The endpoints a user hits to make, view and grab clips never touch
        the Upload-Post key — they work with no key configured at all."""
        job_id = "test-job-gen"
        job_dir = tmp_path / job_id
        job_dir.mkdir()

        # The on-disk metadata the preview/download endpoints read, plus the
        # clip files the download bundle zips up.
        transcript = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "clip one", "words": []},
                {"start": 1.0, "end": 2.0, "text": "clip two", "words": []},
            ]
        }
        (job_dir / "base_metadata.json").write_text(json.dumps({
            "transcript": transcript,
            "shorts": [
                {"start": 0.0, "end": 1.0},
                {"start": 1.0, "end": 2.0},
            ],
        }))
        (job_dir / "base_clip_1.mp4").write_bytes(b"clip one bytes")
        (job_dir / "base_clip_2.mp4").write_bytes(b"clip two bytes")

        jobs[job_id] = {
            "result": {
                "clips": [
                    {"video_url": f"/videos/{job_id}/base_clip_1.mp4"},
                    {"video_url": f"/videos/{job_id}/base_clip_2.mp4"},
                ]
            }
        }

        with pytest.MonkeyPatch().context() as mp:
            mp.delenv("UPLOAD_POST_API_KEY", raising=False)
            # Fail loudly if any of these endpoints try to resolve the key.
            mp.setattr("app.resolve_upload_post",
                       lambda *a, **k: pytest.fail("Upload-Post key resolved"))
            mp.setattr("app.OUTPUT_DIR", str(tmp_path))

            # Preview: word-level captions used by the Remotion editor.
            preview = client.get(f"/api/clip/{job_id}/0/transcript")
            assert preview.status_code == 200, preview.text

            # Download: bundle every clip into one ZIP.
            download = client.get(f"/api/jobs/{job_id}/download-all")
            assert download.status_code == 200, download.text
            assert download.headers["content-type"] == "application/zip"

    def test_publish_proceeds_when_upload_post_key_and_profile_are_given(self, tmp_path):
        """With a key and a profile the request reaches the vendor and returns
        its success payload — the validation gate is not a blanket block."""
        job_id = "test-job-456"
        jobs[job_id] = {
            "result": {
                "clips": [
                    {
                        "video_url": "/videos/test-job-456/clip2.mp4",
                        "title": "Test Clip 2",
                        "video_description_for_tiktok": "Test description 2"
                    }
                ]
            }
        }

        job_dir = tmp_path / job_id
        job_dir.mkdir()
        (job_dir / "clip2.mp4").write_bytes(b"mock video content")

        captured = {}

        def fake_post(url, headers, data, file_path, filename, timeout):
            captured["headers"] = headers
            captured["user"] = data["user"]
            response = MagicMock()
            response.status_code = 202
            response.json.return_value = {"message": "queued"}
            return response

        with pytest.MonkeyPatch().context() as mp:
            mp.delenv("UPLOAD_POST_API_KEY", raising=False)
            mp.setattr("app.OUTPUT_DIR", str(tmp_path))
            mp.setattr("app._post_video_blocking", fake_post)

            response = client.post("/api/social/post", json={
                "job_id": job_id,
                "clip_index": 0,
                "api_key": "test-upload-post-key",
                "user_id": "my-profile",
                "platforms": ["tiktok"]
            })

        assert response.status_code == 200, response.text
        assert response.json() == {"message": "queued"}
        # The resolved key was carried to the vendor, not dropped.
        assert captured["headers"]["Authorization"] == "Apikey test-upload-post-key"
        assert captured["user"] == "my-profile"


if __name__ == "__main__":
    pytest.main([__file__])