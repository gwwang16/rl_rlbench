import logging

class CustomLogger(object):
    def __init__(self, logfile, level=logging.DEBUG):
        self.logger = logging.getLogger()
        self.logger.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.ch1 = logging.FileHandler(logfile, mode='a')
        self.ch1.setLevel(level)
        self.ch1.setFormatter(formatter)
        self.logger.addHandler(self.ch1)

    def info(self, string_msg):
        self.logger.info(string_msg)
    


