from pathlib import Path

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Classify tests by directory while retaining explicit specialized markers."""
    for item in items:
        parts = Path(str(item.path)).parts
        if "unit" in parts:
            item.add_marker(pytest.mark.unit)
        if "native" in parts:
            item.add_marker(pytest.mark.native)
        if "simulator" in parts:
            item.add_marker(pytest.mark.simulator)
