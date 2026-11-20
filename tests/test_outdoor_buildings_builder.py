import numpy as np

from samcloud.bim.core.ifc.writer import write_ifc
from samcloud.drone.contracts import BuildingModel, LocalFrame, Opening
from samcloud.outdoor_model.buildings import build_building_elements, category_counts, to_dxf_entities
from samcloud.outdoor_model.scene import assemble_scene


def _model(status="ok"):
    vertices=np.array([[0,0,0],[4,0,0],[4,3,0],[0,3,0],[0,0,2],[4,0,2],[4,3,2],[0,3,2],[2,0,3],[2,3,3]],float)
    faces=np.array([[0,1,2],[0,2,3],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7],[4,5,8],[4,8,9],[5,6,8],[6,9,8],[6,7,9],[7,4,9]])
    roles=np.array([0,0]+[1]*8+[2]*6)
    return BuildingModel(7,"2.2",vertices,faces,roles,status,reason="synthetic failure" if status=="failed" else "")


def test_gable_building_has_ifc_envelope_openings_and_dxf(tmp_path):
    openings={7:[Opening(7,0,"window",.5,1.5,.7,1.6),Opening(7,0,"door",2.,3.,0.,2.)]}
    elements, qa=build_building_elements([_model()],openings,{})
    assert not qa and category_counts(elements)=={"building":1,"window":1,"door":1}
    dxf=to_dxf_entities(elements,LocalFrame((2600000,1200000,500)))
    assert dxf["S-BLDG"][0]["closed"] and dxf["S-BLDG-ROOF"] and dxf["S-BLDG-WIND"] and dxf["S-BLDG-DOOR"]
    result=write_ifc(assemble_scene(elements,frame=LocalFrame((2600000,1200000,500))),tmp_path/"out.ifc")
    assert result.counts["IfcWall"] and result.counts["IfcRoof"] and result.counts["IfcWindow"] and result.counts["IfcDoor"]


def test_failed_building_is_qa_only():
    elements, qa=build_building_elements([_model("failed")],{}, {})
    assert not elements and qa[0].category=="building"
