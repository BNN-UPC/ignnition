import os
from datetime import datetime
from hashlib import sha256

import loguru
from torch.utils.tensorboard import SummaryWriter


class Logger:
    folder = "logs"
    sterr = True  # logs will be shown in stderr as well

    def __init__(self):
        self.logger = loguru.logger
        self.writer = SummaryWriter(log_dir="logs/tensorflow")  # to log ML-related data
        if not self.sterr:
            self.logger.remove(0)  # remove logs to default channel stderr
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.logger.add(f"{self.folder}/info.log",
                        format="{time} {level} {message}",
                        filter=lambda record: record["level"].name == "INFO")
        self.logger.add(f"{self.folder}/debug.log",
                        format="{time} {level} {message}",
                        filter=lambda record: record["level"].name == "DEBUG")
        self.logger.add(f"{self.folder}/errors.log",
                        format="{time} {level} {message}",
                        filter=lambda record: record["level"].name == "ERROR")

    def __del__(self):
        self.writer.close()

    def log_text(self, text: str, level: str = "INFO", exception: tuple = None) -> None:
        lev = level.lower()
        if lev == "info":
            self.logger.info(text)
        elif lev == "debug":
            if exception:
                self.logger.opt(exception=exception).debug(text)
            else:
                self.logger.debug(text)
        elif lev == "error":
            if exception:
                hash = sha256(str(datetime.now()).encode("utf-8")).hexdigest()[:20]
                message = f"{text} (Error ID: {hash})"
                self.logger.error(message)
                self.logger.opt(exception=exception).debug(message)
            else:
                self.logger.error(text)


logger = Logger()  # find better instantiation -> different config parameters?
