"""Entry point used by the launcher and by the PyInstaller .exe: starts the local UI.

If anything goes wrong the error is shown and written to Terrasort_error.log, and the
window waits for a key press instead of closing.
"""
import os
import sys
import traceback

try:
    if len(sys.argv) < 2 or sys.argv[1] not in ("serve", "classify"):
        sys.argv = [sys.argv[0], "serve"] + sys.argv[1:]
    from terrasort.cli import main

    main()
except SystemExit:
    raise
except BaseException:  # noqa: BLE001
    msg = traceback.format_exc()
    print("\nTerrasort could not start:\n")
    print(msg)
    try:
        base = os.path.dirname(sys.executable if getattr(sys, "frozen", False) else os.path.abspath(__file__))
        with open(os.path.join(base, "Terrasort_error.log"), "w", encoding="utf-8") as f:
            f.write(msg)
        print("(also saved to Terrasort_error.log)")
    except Exception:  # noqa: BLE001
        pass
    input("\nPress Enter to close...")
