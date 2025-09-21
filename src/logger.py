import os
import logging


LOG_FORMATTER = logging.Formatter(
    fmt="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)

# Step 2: Initialize logger
logger = logging.getLogger("SecKnowledge2")

# Step 3: Set up logging level
logger.setLevel(level=getattr(logging, os.getenv("LOG_LEVEL", "info").upper()))

# Step 4: Create and add stream handler
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(LOG_FORMATTER)
logger.addHandler(_stream_handler)