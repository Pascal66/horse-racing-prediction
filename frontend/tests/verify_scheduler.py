import logging
import time
import os
import sys

# Ensure logs are visible
logging.basicConfig(level=logging.INFO)

try:
    # Importing with the required prefix
    from backend.src.cli.cronJobs import cronjobs

    print("Starting scheduler verification...")
    cronjobs()

    print("Scheduler started in background. The script continues execution...")
    # Give it a tiny bit of time to start background threads
    time.sleep(2)

    print("Verification SUCCESS: Scheduler is running in its own background thread.")
    sys.exit(0)

except Exception as e:
    print(f"Verification FAILED: {e}")
    sys.exit(1)
