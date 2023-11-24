import logging
from logging import getLogger

################################################################
# Set logger
simple_formatter = logging.Formatter("[%(name)s] %(message)s")
complex_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] - %(message)s"
)

app_logger = getLogger()
handler = logging.StreamHandler()
handler.setFormatter(complex_formatter)
app_logger.addHandler(handler)
app_logger.setLevel(logging.INFO)
################################################################
