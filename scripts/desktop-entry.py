import os
import sys
import tempfile
import traceback
from pathlib import Path

if __name__ == "__main__":
    try:
        from atlas.__main__ import main

        main()
    except Exception:
        root = Path(
            os.environ.get("ATLAS_DATA_DIR")
            or Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "AtlasScout"
        )
        root.mkdir(parents=True, exist_ok=True)
        report = root / "startup-error.log"
        report.write_text(traceback.format_exc(), encoding="utf-8")
        if (
            not any(
                flag in sys.argv
                for flag in (
                    "--smoke-test",
                    "--ai-smoke-test",
                    "--formats-smoke-test",
                    "--large-files-smoke-test",
                    "--ecw-index-smoke-test",
                    "--similarity-smoke-test",
                    "--precision-smoke-test",
                )
            )
            and os.name == "nt"
        ):
            import ctypes

            ctypes.windll.user32.MessageBoxW(
                None, f"Atlas could not start. Error details:\n{report}", "Atlas Scout", 16
            )
        raise SystemExit(1) from None
