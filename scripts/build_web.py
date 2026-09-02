"""Build the deployable static web version (pygbag/WASM) of the alien battle
game, ready to upload to Vercel (or any static host) as-is.

Two steps, because `pygbag --build` alone is not enough for static hosting:
  1. `pygbag --build web/dqn_alien_battle_web` compiles/packs the app into
     web/dqn_alien_battle_web/build/web/ (index.html + the app archive).
  2. Vendor the pygame-ce wasm wheel into build/web/cdn/cp312/ ourselves.
     pygbag's generated page fetches extra packages (anything the app
     imports beyond the interpreter itself — here, pygame-ce) from a
     same-origin relative path (/cdn/cp312/<wheel>), not the real CDN it was
     itself loaded from. That path only gets served automatically by
     pygbag's own local dev server, which a plain static host (Vercel, S3,
     GitHub Pages, ...) doesn't replicate — reproduced firsthand during
     development as a silent 404 with the game frozen on a blank canvas.
     So we fetch the real wheel from the CDN once and vendor a copy at the
     exact path the runtime expects, making the deployed output
     self-contained.

Note: this web build deliberately has no `numpy` (or `torch`/`gymnasium`)
anywhere in its code — see pure_dqn.py and battle_logic.py's docstrings.
pygbag auto-detects `import numpy` and tries to dynamically install/compile
it at runtime, which was reproduced to hang indefinitely. Don't reintroduce
a numpy import into the web build without re-testing that path carefully.

Usage:
    python scripts/build_web.py
Output ready to deploy: web/dqn_alien_battle_web/build/web/
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

APP_DIR = Path("web/dqn_alien_battle_web")
BUILD_DIR = APP_DIR / "build" / "web"
PYGAME_WHEEL = "pygame_ce-2.5.7-cp312-cp312-wasm32_bi_emscripten.whl"
CDN_BASE = "https://pygame-web.github.io/cdn/cp312/"


def main() -> None:
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR, ignore_errors=True)

    subprocess.run(
        [sys.executable, "-m", "pygbag", "--build", str(APP_DIR)],
        check=True,
    )

    wheel_dir = BUILD_DIR / "cdn" / "cp312"
    wheel_dir.mkdir(parents=True, exist_ok=True)
    wheel_path = wheel_dir / PYGAME_WHEEL
    print(f"Vendoring {PYGAME_WHEEL} for static hosting...")
    urllib.request.urlretrieve(CDN_BASE + PYGAME_WHEEL, wheel_path)
    print(f"  -> {wheel_path} ({wheel_path.stat().st_size:,} bytes)")

    print(f"\nDone. Deploy {BUILD_DIR}/ as a static site (e.g. `vercel deploy {BUILD_DIR}`).")


if __name__ == "__main__":
    main()
