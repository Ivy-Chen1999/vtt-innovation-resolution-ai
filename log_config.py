import logging
import os


os.makedirs("logs", exist_ok = True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),             
        logging.FileHandler("logs/app.log")
    ]
)

logger = logging.getLogger(__name__) 