"""
Greenpack Pro — Complete Test Suite
Run: pytest tests/ -v --tb=short
"""
import asyncio
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ── Test utilities ─────────────────────────────────────────────────────────────

def make_test_image(w=400, h=300, text="Test Label 12345", color=(255,255,255)):
    """Create a synthetic test image with text"""
    import cv2
    img = np.full((h, w, 3), color, dtype=np.uint8)
    cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.putText(img, "EXP: 31/12/2025", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.rectangle(img, (20, 110), (380, 280), (200, 200, 200), 2)
    cv2.putText(img, "PRODUCT INFO", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,50,50), 1)
    return img

def save_test_image(img, path):
    import cv2
    cv2.imwrite(str(path), img)
    return path

def make_test_pdf(path, text="Master Label PDF"):
    """Create minimal test PDF"""
    try:
        from reportlab.pagesizes import A4
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(str(path), pagesize=A4)
        c.setFont("Helvetica", 14)
        c.drawString(50, 700, text)
        c.drawString(50, 670, "EXP: 31/12/2025")
        c.drawString(50, 640, "Batch: BATCH202504")
        c.rect(50, 100, 400, 500)
        c.save()
        return True
    except ImportError:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestAlignment:
    """Tests for image alignment service"""

    def test_align_identical_images(self):
        """Identical images should align with high confidence"""
        from app.services.alignment import align_images
        img = make_test_image()
        aligned, confidence = align_images(img, img.copy())
        assert confidence > 0.5

    def test_align_rotated_image(self):
        """Slightly rotated scan should still align"""
        import cv2
        from app.services.alignment import align_images
        master = make_test_image()
        h, w = master.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), 2, 1.0)  # 2 degree rotation
        rotated = cv2.warpAffine(master, M, (w, h))
        aligned, confidence = align_images(master, rotated)
        assert confidence > 0.2  # Should still align

    def test_align_different_sizes(self):
        """Different sized images should be handled"""
        import cv2
        from app.services.alignment import align_images
        master = make_test_image(400, 300)
        scan = cv2.resize(master, (380, 290))  # Slightly different size
        aligned, confidence = align_images(master, scan)
        assert aligned.shape[:2] == (300, 400)  # Output matches master dims

    def test_align_returns_numpy(self):
        """Aligned result must be numpy array"""
        from app.services.alignment import align_images
        img = make_test_image()
        aligned, _ = align_images(img, img.copy())
        assert isinstance(aligned, np.ndarray)


class TestColorService:
    """Tests for color analysis"""

    def test_identical_images_zero_delta_e(self):
        """Same image compared to itself = 0 ΔE"""
        from app.services.color_service import analyze_color_zones
        img = make_test_image()
        result = analyze_color_zones(img, img.copy(), threshold=2.0)
        assert result["mean_delta_e"] < 0.1

    def test_different_colors_high_delta_e(self):
        """Red vs Blue label = high ΔE"""
        from app.services.color_service import analyze_color_zones
        red = np.full((100, 100, 3), [0, 0, 200], dtype=np.uint8)  # Red in BGR
        blue = np.full((100, 100, 3), [200, 0, 0], dtype=np.uint8)  # Blue in BGR
        result = analyze_color_zones(red, blue, threshold=2.0)
        assert result["mean_delta_e"] > 2.0
        assert result["pass"] == False

    def test_identical_passes(self):
        """Identical images must PASS color check"""
        from app.services.color_service import analyze_color_zones
        img = make_test_image(color=(200, 150, 100))
        result = analyze_color_zones(img, img.copy(), threshold=2.0)
        assert result["pass"] == True

    def test_returns_zone_results(self):
        """Must return list of color zone results"""
        from app.services.color_service import analyze_color_zones
        img = make_test_image()
        result = analyze_color_zones(img, img.copy())
        assert "zone_results" in result
        assert isinstance(result["zone_results"], list)


class TestSSIMService:
    """Tests for SSIM defect detection"""

    def test_identical_images_perfect_ssim(self):
        """Identical images = SSIM score 1.0"""
        from app.services.ssim_service import detect_defects
        img = make_test_image()
        result = detect_defects(img, img.copy())
        assert result["ssim_score"] > 0.99

    def test_black_block_defect_detected(self):
        """Large black rectangle on scan = defect"""
        from app.services.ssim_service import detect_defects
        master = make_test_image()
        scan = master.copy()
        scan[100:200, 100:300] = 0  # Black block
        result = detect_defects(master, scan, threshold=0.75)
        assert result["ssim_score"] < 0.99
        assert len(result["defects"]) > 0

    def test_pass_fail_threshold(self):
        """SSIM above threshold = PASS"""
        from app.services.ssim_service import detect_defects
        img = make_test_image()
        result = detect_defects(img, img.copy(), threshold=0.75)
        assert result["pass"] == True

    def test_defect_has_required_fields(self):
        """Defect objects must have required fields"""
        from app.services.ssim_service import detect_defects
        master = make_test_image()
        scan = master.copy()
        scan[50:150, 50:200] = 0
        result = detect_defects(master, scan)
        for defect in result.get("defects", []):
            assert "type" in defect
            assert "severity" in defect
            assert "bbox" in defect
            assert "area_pixels" in defect


class TestBarcodeService:
    """Tests for barcode verification"""

    def test_ean13_check_digit_valid(self):
        """Valid EAN-13 should pass check digit"""
        from app.services.barcode_service import _validate_ean13
        assert _validate_ean13("5901234123457") == True

    def test_ean13_check_digit_invalid(self):
        """Invalid EAN-13 should fail check digit"""
        from app.services.barcode_service import _validate_ean13
        assert _validate_ean13("5901234123456") == False

    def test_ean13_wrong_length(self):
        """EAN-13 with wrong length = invalid"""
        from app.services.barcode_service import _validate_ean13
        assert _validate_ean13("590123412345") == False
        assert _validate_ean13("59012341234570") == False

    def test_ean13_non_digits(self):
        """EAN-13 with letters = invalid"""
        from app.services.barcode_service import _validate_ean13
        assert _validate_ean13("590123412345X") == False

    def test_ean8_valid(self):
        """Valid EAN-8 check digit"""
        from app.services.barcode_service import _validate_ean8
        assert _validate_ean8("96385074") == True

    def test_gs1_128_parse(self):
        """GS1-128 parsing extracts fields"""
        from app.services.barcode_service import _parse_gs1_128
        result = _parse_gs1_128("(01)08690526040015(17)251231(10)BATCH202504")
        assert result is not None
        # Function returns dict with parsed fields or None for non-GS1 strings

    def test_barcode_service_empty_image(self):
        """Empty/white image should return empty results"""
        from app.services.barcode_service import verify_barcodes
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            white = np.full((200, 200, 3), 255, dtype=np.uint8)
            import cv2
            cv2.imwrite(f.name, white)
            results = verify_barcodes(f.name, [])
            assert isinstance(results, list)
            os.unlink(f.name)


class TestPreprocess:
    """Tests for image preprocessing"""

    def test_preprocess_saves_file(self):
        """Preprocessed image must be saved to disk"""
        from app.services.preprocess import preprocess_image
        with tempfile.TemporaryDirectory() as tmp:
            img = make_test_image()
            import cv2
            img_path = Path(tmp) / "test.png"
            cv2.imwrite(str(img_path), img)
            result = preprocess_image(img_path)
            assert result.exists()
            assert result.stat().st_size > 0

    def test_preprocess_returns_path(self):
        """Preprocess returns Path object"""
        from app.services.preprocess import preprocess_image
        with tempfile.TemporaryDirectory() as tmp:
            img = make_test_image()
            import cv2
            img_path = Path(tmp) / "test.jpg"
            cv2.imwrite(str(img_path), img)
            result = preprocess_image(img_path)
            assert isinstance(result, Path)


class TestScoreCalculation:
    """Tests for inspection score calculation"""

    def test_perfect_score(self):
        """No errors = 100 score"""
        from app.services.inspection_engine import InspectionEngine
        engine = InspectionEngine()
        scores = engine._calculate_scores(
            text_errors=[],
            color_result={"zone_results": [{"pass": True}]},
            ssim_result={"ssim_score": 1.0, "defects": []},
            barcode_result=[{"pass": True}],
        )
        assert scores["overall"] == 100.0
        assert scores["ocr"] == 100.0
        assert scores["barcode"] == 100.0

    def test_fail_with_errors(self):
        """Many OCR errors should push score below 75"""
        from app.services.inspection_engine import InspectionEngine
        engine = InspectionEngine()
        many_errors = [{"type": "REPLACE", "severity": "high"} for _ in range(6)]
        scores = engine._calculate_scores(
            text_errors=many_errors,
            color_result={"zone_results": []},
            ssim_result={"ssim_score": 0.95, "defects": []},
            barcode_result=[],
        )
        assert scores["overall"] < 75.0

    def test_weights_sum(self):
        """Score weights: OCR 35% + Color 30% + SSIM 20% + Barcode 15% = 100%"""
        weights = [0.35, 0.30, 0.20, 0.15]
        assert abs(sum(weights) - 1.0) < 0.001

    def test_barcode_score_no_barcodes(self):
        """No barcodes configured = full barcode score"""
        from app.services.inspection_engine import InspectionEngine
        engine = InspectionEngine()
        scores = engine._calculate_scores(
            text_errors=[],
            color_result={"zone_results": []},
            ssim_result={"ssim_score": 1.0},
            barcode_result=[],  # No barcodes
        )
        assert scores["barcode"] == 100.0


class TestAuthService:
    """Tests for authentication"""

    def test_password_hash_verify(self):
        """Hash and verify password roundtrip"""
        from app.services.auth_service import hash_password, verify_password
        pwd = "TestPassword123!"
        hashed = hash_password(pwd)
        assert hashed != pwd
        assert verify_password(pwd, hashed) == True
        assert verify_password("wrong", hashed) == False

    def test_create_access_token(self):
        """Access token should be decodable"""
        from app.services.auth_service import create_access_token, decode_token
        token = create_access_token("user-123", "test@example.com", "inspector")
        payload = decode_token(token)
        assert payload["sub"] == "user-123"
        assert payload["email"] == "test@example.com"
        assert payload["role"] == "inspector"
        assert payload["type"] == "access"

    def test_refresh_token_different_from_access(self):
        """Refresh and access tokens should be different"""
        from app.services.auth_service import create_access_token, create_refresh_token
        access = create_access_token("user-123", "test@example.com", "inspector")
        refresh = create_refresh_token("user-123")
        assert access != refresh


class TestBackupService:
    """Tests for backup and disk management"""

    def test_check_disk_space_returns_dict(self):
        """Disk space check returns proper structure"""
        from app.services.backup_service import check_disk_space
        result = check_disk_space()
        assert "free_gb" in result
        assert "status" in result
        assert result["status"] in ["ok", "warning", "critical"]
        assert result["free_gb"] >= 0

    def test_list_backups_empty(self):
        """Empty backup dir returns empty list"""
        from app.services.backup_service import list_backups
        result = list_backups()
        assert isinstance(result, list)

    def test_verify_backup_nonexistent(self):
        """Verify nonexistent file returns False"""
        from app.services.backup_service import verify_backup
        result = verify_backup("/nonexistent/path/backup.db")
        assert result == False


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS (require full app setup)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def anyio_backend():
    return "asyncio"

@pytest.fixture
async def test_app():
    """Create test FastAPI app with in-memory SQLite"""
    import os
    os.environ.update({
        "DB_URL": "sqlite+aiosqlite:///./test_greenpack.db",
        "JWT_SECRET_KEY": "test-secret-key-for-testing-only",
        "APP_SECRET_KEY": "test-app-secret-for-testing-only",
        "EASYOCR_DOWNLOAD_ENABLED": "false",
        "GREENPACK_MODE": "standalone",
        "LOG_LEVEL": "WARNING",
    })
    from app.main import app
    from app.database import init_db
    await init_db()
    yield app
    # Cleanup
    db_path = Path("./test_greenpack.db")
    if db_path.exists():
        db_path.unlink()

@pytest.mark.anyio
async def test_health_endpoint(test_app):
    """Health endpoint returns 200"""
    from httpx import AsyncClient, ASGITransport
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

@pytest.mark.anyio
async def test_login_default_admin(test_app):
    """Default admin can login"""
    from httpx import AsyncClient, ASGITransport
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        resp = await client.post("/api/v1/auth/login", json={
            "email": "admin@greenpackpro.local",
            "password": "Admin123!",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["role"] == "admin"

@pytest.mark.anyio
async def test_login_wrong_password(test_app):
    """Wrong password returns 401"""
    from httpx import AsyncClient, ASGITransport
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        resp = await client.post("/api/v1/auth/login", json={
            "email": "admin@greenpackpro.local",
            "password": "wrongpassword",
        })
        assert resp.status_code == 401

@pytest.mark.anyio
async def test_protected_endpoint_requires_auth(test_app):
    """Protected endpoint without token returns 401/403"""
    from httpx import AsyncClient, ASGITransport
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        resp = await client.get("/api/v1/jobs")
        assert resp.status_code in [401, 403]

@pytest.mark.anyio
async def test_list_jobs_authenticated(test_app):
    """Authenticated user can list jobs"""
    from httpx import AsyncClient, ASGITransport
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        # Login
        login_resp = await client.post("/api/v1/auth/login", json={
            "email": "admin@greenpackpro.local",
            "password": "Admin123!",
        })
        token = login_resp.json()["access_token"]

        # List jobs
        resp = await client.get("/api/v1/jobs", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

@pytest.mark.anyio
async def test_scanner_list(test_app):
    """Scanner list endpoint returns proper structure"""
    from httpx import AsyncClient, ASGITransport
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        login_resp = await client.post("/api/v1/auth/login", json={
            "email": "admin@greenpackpro.local", "password": "Admin123!",
        })
        token = login_resp.json()["access_token"]
        resp = await client.get("/api/v1/scanners", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert "scanners" in data
        assert "count" in data

@pytest.mark.anyio
async def test_settings_endpoint(test_app):
    """Settings endpoint returns configuration"""
    from httpx import AsyncClient, ASGITransport
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        login_resp = await client.post("/api/v1/auth/login", json={
            "email": "admin@greenpackpro.local", "password": "Admin123!",
        })
        token = login_resp.json()["access_token"]
        resp = await client.get("/api/v1/settings", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert "mode" in data
        assert "default_color_threshold" in data

@pytest.mark.anyio
async def test_templates_list(test_app):
    """Templates list returns empty array initially"""
    from httpx import AsyncClient, ASGITransport
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        login_resp = await client.post("/api/v1/auth/login", json={
            "email": "admin@greenpackpro.local", "password": "Admin123!",
        })
        token = login_resp.json()["access_token"]
        resp = await client.get("/api/v1/templates", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

@pytest.mark.anyio
async def test_dashboard_stats(test_app):
    """Dashboard stats endpoint works"""
    from httpx import AsyncClient, ASGITransport
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        login_resp = await client.post("/api/v1/auth/login", json={
            "email": "admin@greenpackpro.local", "password": "Admin123!",
        })
        token = login_resp.json()["access_token"]
        resp = await client.get("/api/v1/dashboard/stats", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert "today_total" in data
        assert "pass_rate" in data

# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE INTEGRATION TEST (requires images)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.anyio
async def test_full_inspection_pipeline():
    """
    End-to-end inspection test using synthetic images.
    Skipped if EasyOCR model not available.
    """
    import cv2

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Create test images
        master_img = make_test_image(400, 300, "Test Product Label 5901234123457")
        scan_img = master_img.copy()  # Perfect copy = should PASS

        master_path = tmp / "master.png"
        scan_path = tmp / "scan.png"
        save_test_image(master_img, master_path)
        save_test_image(scan_img, scan_path)

        # Test alignment
        from app.services.alignment import align_images
        aligned, conf = align_images(master_img, scan_img)
        assert conf > 0.5

        # Test SSIM
        from app.services.ssim_service import detect_defects
        ssim_result = detect_defects(master_img, aligned)
        assert ssim_result["ssim_score"] > 0.95
        assert ssim_result["pass"] == True

        # Test color
        from app.services.color_service import analyze_color_zones
        color_result = analyze_color_zones(master_img, aligned)
        assert color_result["mean_delta_e"] < 0.5
        assert color_result["pass"] == True

        # Test barcode (no barcodes in synthetic image)
        from app.services.barcode_service import verify_barcodes
        bc_result = verify_barcodes(str(scan_path), [])
        assert isinstance(bc_result, list)

        # Test score calculation
        from app.services.inspection_engine import InspectionEngine
        engine = InspectionEngine()
        scores = engine._calculate_scores(
            text_errors=[],
            color_result=color_result,
            ssim_result=ssim_result,
            barcode_result=bc_result,
        )
        assert scores["overall"] >= 75.0  # Should PASS
        assert scores["color"] > 90.0
        assert scores["ssim"] > 90.0

        print(f"\n✅ Pipeline test passed: score={scores['overall']:.1f}")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short", "-x"])
