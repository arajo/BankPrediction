import logging
from logging import getLogger


class Logger:
    def __init__(self):
        self.simple_formatter = logging.Formatter("[%(name)s] %(message)s")
        self.complex_formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] - %(message)s")
        self.log_level = logging.INFO

    def get_logger(self):
        logger = getLogger("streamlit-logger")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(self.complex_formatter)
            logger.addHandler(handler)
            logger.setLevel(self.log_level)
        return logger


app_logger = Logger().get_logger()
app_logger.propagate = False
