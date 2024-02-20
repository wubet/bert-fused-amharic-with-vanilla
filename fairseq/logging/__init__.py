import logging
from . import meters
from . import metrics
from . import progress_bar

# Configure the root logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = True

# Create a stream handler to output to console
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Add the stream handler to the logger
logger.addHandler(stream_handler)