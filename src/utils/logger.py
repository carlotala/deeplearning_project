"""
src/utils/logger.py.

- This file contains the definition of a general and standard logger for
    logging in the project's code.

"""
import logging
import os
import time

basepath = os.path.dirname(__file__)


# with date --> new log file each day
# A timestamp is added to have a new logger each day
def get_logger() -> logging.Logger:
    """
    Initialize a logger to print messages to a log file.

    Returns
    -------
    logging.Logger
        Logger object.
    """
    PROJECT_ROOT = os.path.abspath(os.path.join(basepath, "..", ".."))
    log_path = os.path.join(PROJECT_ROOT, "outputs", "logs")
    error_log_filename = os.path.join(
        log_path, time.strftime("%Y%m%d_") + "error_logs.log"
    )
    all_log_filename = os.path.join(log_path, time.strftime("%Y%m%d_") + "all_logs.log")

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # Define the parameters to display and the format
    mformat = (
        "%(asctime)s | %(name)s | %(levelname)s | %(filename)s |"
        " @function %(funcName)s | line %(lineno)s | %(message)s"
    )
    datefmt = "%Y-%m-%d %H:%M%S"

    # Create a logger
    logger = logging.getLogger(name="waste-classification")
    logger.setLevel(logging.DEBUG)

    # Create "console handler" and define the information level
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(mformat, datefmt=datefmt))
    logger.addHandler(handler)

    # Create "error file handler" and define the error level
    fileHandler = logging.FileHandler(filename=error_log_filename)
    fileHandler.setFormatter(logging.Formatter(mformat, datefmt=datefmt))
    fileHandler.setLevel(level=logging.ERROR)

    logger.addHandler(fileHandler)

    # Create "debug file handler" and define the debug level
    fileHandler = logging.FileHandler(filename=all_log_filename)
    fileHandler.setFormatter(logging.Formatter(mformat, datefmt=datefmt))
    fileHandler.setLevel(level=logging.INFO)

    logger.addHandler(fileHandler)

    return logger


logger_all = get_logger()
