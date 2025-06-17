import os
import sys
import logging
import io  

# Safe override only if needed
# if os.name == "nt" and hasattr(sys.stdout, "fileno"):
#     try:
#         sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
#     except io.UnsupportedOperation:
#         pass

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("mlProjectLogger")
