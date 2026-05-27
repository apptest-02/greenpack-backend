"""
Greenpack Pro — Scanner Service
TWAIN/WIA scanner integration via Dynamic Web TWAIN Service.
Webcam fallback via OpenCV.
"""
import logging
from pathlib import Path
from typing import Optional
import cv2
import base64
import uuid

from app.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()

DWT_HOST = settings.dwt_service_url


def list_scanners() -> list[dict]:
    """Return list of connected TWAIN/WIA scanners"""
    try:
        from dynamsoftservice import ScannerController, ScannerType
        ctrl = ScannerController()
        scanners = ctrl.getDevices(
            DWT_HOST,
            ScannerType.TWAINSCANNER | ScannerType.TWAINX64SCANNER | ScannerType.WIASCANNER,
        )
        log.info(f"Found {len(scanners)} scanner(s)")
        return [{"name": s.get("name", "Unknown"), "device": s.get("device"), "type": "TWAIN"} for s in scanners]
    except ImportError:
        log.warning("twain-wia-sane-scanner not installed — scanner listing unavailable")
        return _list_webcams()
    except Exception as e:
        log.warning(f"Scanner discovery failed (DWT Service may not be running): {e}")
        return _list_webcams()


def _list_webcams() -> list[dict]:
    """Fallback: list USB webcams via OpenCV"""
    webcams = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            webcams.append({
                "name": f"USB Camera {i}",
                "device": f"webcam:{i}",
                "type": "WEBCAM",
            })
            cap.release()
    return webcams


def capture_from_scanner(
    device_id: str,
    resolution: int = 300,
    output_dir: Optional[str] = None,
) -> str:
    """
    Trigger scan from TWAIN/WIA scanner.
    Returns path to saved image.
    """
    out_dir = Path(output_dir or settings.temp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if device_id.startswith("webcam:"):
        # Webcam capture
        cam_index = int(device_id.split(":")[1])
        return capture_from_webcam(cam_index, str(out_dir))

    try:
        from dynamsoftservice import ScannerController
        ctrl = ScannerController()
        params = {
            "license": settings.dwt_license_key,
            "device": device_id,
            "config": {
                "IfShowUI": False,
                "PixelType": 2,  # 2=RGB Color
                "Resolution": resolution,
                "IfFeederEnabled": False,
                "IfDuplexEnabled": False,
            },
        }
        job = ctrl.createJob(DWT_HOST, params)
        job_id = job.get("jobuid", "")
        if not job_id:
            raise RuntimeError("Scanner job creation failed — check DWT Service is running")

        images = ctrl.getImageFiles(DWT_HOST, job_id, str(out_dir))
        if not images:
            raise RuntimeError("No image returned from scanner")

        log.info(f"Scanner capture saved: {images[0]}")
        return images[0]

    except ImportError:
        log.error("twain-wia-sane-scanner not installed")
        raise RuntimeError("Scanner driver not installed. Please install Dynamic Web TWAIN Service.")
    except Exception as e:
        log.error(f"Scanner capture failed: {e}")
        raise RuntimeError(f"Scanner error: {str(e)}")


def capture_from_webcam(device_index: int = 0, output_dir: str = None) -> str:
    """Capture from USB webcam via OpenCV"""
    out_dir = Path(output_dir or settings.temp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam at index {device_index}")

    # Request high resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

    # Capture multiple frames and use the last (camera stabilizes)
    for _ in range(5):
        ret, frame = cap.read()

    cap.release()

    if not ret or frame is None:
        raise RuntimeError("Webcam capture failed — no frame received")

    out_path = out_dir / f"webcam_{uuid.uuid4().hex[:8]}.png"
    cv2.imwrite(str(out_path), frame)
    log.info(f"Webcam capture: {out_path} ({frame.shape[1]}×{frame.shape[0]})")
    return str(out_path)


def get_camera_preview_frame(device_id: str) -> Optional[str]:
    """
    Get single preview frame for live preview UI.
    Returns base64 JPEG or None.
    """
    try:
        if device_id.startswith("webcam:"):
            cam_index = int(device_id.split(":")[1])
        else:
            cam_index = 0

        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buffer).decode("utf-8")
    except Exception as e:
        log.error(f"Preview frame error: {e}")
        return None
