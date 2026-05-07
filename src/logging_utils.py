from __future__ import annotations

import logging

try:
    from loguru import logger
except Exception:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    logger = logging.getLogger("finance_rag")
