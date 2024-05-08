import os
import pytest

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging
test_path = os.path.dirname(os.path.realpath(__file__))


def _create_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)
    return logger


def run_all_tests():
    logger = _create_logger()
    pytest.main([test_path, "-s"])
    logger.setLevel(logging.INFO)


if __name__ == "__main__":
    run_all_tests()
