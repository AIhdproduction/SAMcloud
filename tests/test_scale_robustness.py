"""Scale robustness of the indoor chain: decisions follow geometry, not one metric threshold crossing."""

from __future__ import annotations

import numpy as np
import pytest

from samcloud.bim.scene_builder import _build_joins, _Wall, load_scene_config


def _wall(name: str, p0: tuple[float, float], p1: tuple[float, float], thickness: float) -> _Wall:
    return _Wall(name, np.asarray(p0, dtype=float), np.asarray(p1, dtype=float), thickness, "interior", [name])


@pytest.mark.parametrize("scale", [0.984, 0.9891, 0.994, 1.0, 1.01])
def test_lintel_end_is_never_joined_through_the_wall_it_touches(scale: float) -> None:
    """A lintel ending at the next collinear wall piece must not T-join the wall behind that piece (v12 O005).

    The cross wall's centreline lies 0.40 m beyond the lintel end, exactly at the 0.40 m join reach; whether the join
    formed used to depend on a 1 % scale change.
    """
    cfg = load_scene_config()
    reach = float(cfg["walls"]["join"]["reach_m"])
    s = scale
    lintel = _wall("L_O005", (0.0, 0.0), (0.0, 0.67 * s), 0.30 * s)
    host = _wall("W015", (0.0, 0.67 * s), (0.0, 2.0 * s), 0.29 * s)  # continues the lintel line
    cross = _wall("W029", (0.144 * s, 0.67 * s + reach), (1.6 * s, 0.67 * s + reach), 0.29 * s)  # ends against W015
    joins = _build_joins([lintel, host, cross], cfg)
    assert not any(kind == "T" and a == "L_O005" and b == "W029" for kind, a, _, b, _ in joins)
    assert any(kind == "T" and a == "W029" and b == "W015" for kind, a, _, b, _ in joins)


def test_end_already_inside_the_path_wall_keeps_its_t_join() -> None:
    """Two collinear walls meeting a cross wall from both sides: both T-join it, even when one reaches past its axis."""
    cfg = load_scene_config()
    cross = _wall("W017", (0.0, -1.3), (0.0, 1.2), 0.20)
    east = _wall("W011", (-0.05, 0.0), (1.0, 0.0), 0.255)  # starts 5 cm past the cross wall axis
    west = _wall("W026", (-0.05, 0.0), (-0.65, 0.0), 0.29)
    joins = _build_joins([cross, east, west], cfg)
    assert any(kind == "T" and a == "W026" and b == "W017" for kind, a, _, b, _ in joins)
    assert any(kind == "T" and a == "W011" and b == "W017" for kind, a, _, b, _ in joins)
