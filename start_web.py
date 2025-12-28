#!/usr/bin/env python
import os
import subprocess
import sys

if __name__ == "__main__":
    # Start gunicorn with app.py
    cmd = ["gunicorn", "--bind", "0.0.0.0:5000", "--reuse-port", "--reload", "app:app"]
    try:
        process = subprocess.run(cmd)
        sys.exit(process.returncode)
    except KeyboardInterrupt:
        print("Web server stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting web server: {e}")
        sys.exit(1)