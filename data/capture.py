"""Capture screenshots on left-click. Right-click to stop."""

import sys
import time
from datetime import datetime
from pathlib import Path

import mss
from PIL import Image
from pynput import mouse

OUTPUT_DIR = Path(__file__).resolve().parent / "screenshots"


def _take_screenshot() -> Path:
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        raw = sct.grab(monitor)
        img = Image.frombytes("RGB", (raw.width, raw.height), raw.rgb)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dest = OUTPUT_DIR / f"{stamp}.jpg"
    img.save(dest, format="JPEG", quality=85)
    return dest


def on_click(x, y, button, pressed):
    if not pressed:
        return
    if button == mouse.Button.right:
        print("Right-click detected, stopping.")
        return False
    if button == mouse.Button.left:
        path = _take_screenshot()
        print(f"Saved {path.name}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Left-click to capture screenshots to {OUTPUT_DIR}/")
    print("Right-click to stop.")
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()


if __name__ == "__main__":
    main()
