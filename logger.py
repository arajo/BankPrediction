import logging
from logging import getLogger


class Logger:
    def __init__(self):
        self.simple_formatter = logging.Formatter("[%(name)s] %(message)s")
        self.complex_formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] - %(message)s")
        self.log_level = logging.INFO

    def get_logger(self):
        app_logger = getLogger()
        handler = self.set_handler()
        app_logger.addHandler(handler)
        app_logger.setLevel(self.log_level)
        return app_logger

    def set_handler(self, ):
        handler = logging.StreamHandler()
        handler.setFormatter(self.complex_formatter)
        return handler


app_logger = Logger().get_logger()
app_logger.propagate = False
