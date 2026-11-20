import ezdxf

from samcloud.drone.contracts import LocalFrame
from samcloud.dxf.writer import build_dxf_document
from samcloud.outdoor_model.scene import assemble_dxf, assemble_scene


def test_dxf_defaults_to_lv95_and_can_use_local_origin():
    frame=LocalFrame((2600000,1200000,500))
    result=assemble_dxf({"x":{"S-BLDG":[{"type":"lwpolyline","points":[[2600000,1200000,500],[2600001,1200000,500],[2600001,1200001,500]],"closed":True,"elevation":500}]}},frame)
    full=build_dxf_document(result,coordinates="local")
    assert next(iter(full.modelspace())).dxf.layer=="S-BLDG"
    local=build_dxf_document(result,coordinates="local",local_origin=True,origin_offset=frame.origin_lv95)
    assert tuple(local.header["$INSBASE"])==frame.origin_lv95


def test_scene_is_outdoor_lv95():
    scene=assemble_scene(frame=LocalFrame((2600000,1200000,500)))
    assert scene.settings.scope.value=="outdoor_only" and scene.settings.crs.epsg==2056
