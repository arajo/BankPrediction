import logging
from logging import getLogger

################################################################
# Set logger
app_logger = getLogger()
app_logger.addHandler(logging.StreamHandler())
app_logger.setLevel(logging.INFO)
################################################################
