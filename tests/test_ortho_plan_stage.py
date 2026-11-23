"""An optional refinement stage of the indoor plan never costs the whole plan (``samcloud.ortho.plan._plan_stage``)."""

from __future__ import annotations

import pytest

from samcloud.dxf.cancel import Cancelled
from samcloud.ortho.plan import _plan_stage


def _layers() -> dict[str, list[dict[str, object]]]:
    return {
        "A-EINB": [{"type": "polyline", "points": [[1.03, 2.01], [3.12, 2.02], [3.11, 2.66], [1.02, 2.64]], "closed": True, "_einb_id": 7}],
        "A-ANNO": [{"type": "text", "text": "Küche", "point": [2.07, 2.33]}],
    }


def test_a_failing_stage_puts_the_layers_back_and_returns_the_default() -> None:
    layers = _layers()
    einb = layers["A-EINB"]  # other code holds the list itself: it must keep its identity
    failures: dict[str, int] = {}

    def stage() -> dict[str, int]:
        layers["A-EINB"][0]["points"][0][0] = 9.9          # a half-done change inside an entity ...
        layers["A-EINB"].append({"type": "polyline", "points": [], "closed": True})
        del layers["A-ANNO"]                               # ... a layer lost ...
        layers["A-NEW"] = [{"type": "text"}]               # ... and one added, then the stage fails
        raise ZeroDivisionError("degenerate leg")

    assert _plan_stage(failures, "kitchen_cad_layout", stage, default={}, layers=layers) == {}
    assert failures == {"kitchen_cad_layout": 1}
    assert layers == _layers() and layers["A-EINB"] is einb


def test_a_stage_that_works_returns_its_result_and_keeps_its_changes() -> None:
    layers = _layers()
    failures: dict[str, int] = {}

    def stage() -> dict[str, int]:
        layers["A-EINB"][0]["_kitchen_run"] = "L"
        return {"kitchen_legs": 2}

    assert _plan_stage(failures, "kitchen_cad_layout", stage, default={}, layers=layers) == {"kitchen_legs": 2}
    assert failures == {} and layers["A-EINB"][0]["_kitchen_run"] == "L"
    assert _plan_stage(failures, "door_bounded_rooms", lambda: (["room"], {"n": 1}), default=([], {})) == (["room"], {"n": 1})


def test_cancel_and_memory_errors_are_not_swallowed() -> None:
    for error in (Cancelled("stop"), MemoryError()):
        def stage(error: BaseException = error) -> None:
            raise error

        with pytest.raises(type(error)):
            _plan_stage({}, "door_bounded_rooms", stage, default=([], {}))
