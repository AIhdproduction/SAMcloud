import numpy as np

from samcloud.drone.contracts import FENCE, SITE_WALL, LocalFrame, PointSet
from samcloud.outdoor_model.site_walls import build_site_elements, fence_features, site_wall_features, to_dxf_entities


def test_wall_and_fence_features_and_dxf():
    rng=np.random.default_rng(3); x=np.repeat(np.linspace(0,5,600),3); offset=np.tile(np.array([-.125,0,.125]),600)
    wall=np.c_[x,offset,rng.uniform(0,1.5,len(x))]; fence=np.c_[np.linspace(0,5,600),np.full(600,2.),rng.uniform(0,1.2,600)]
    ps=PointSet(np.r_[wall,fence],np.r_[np.full(len(wall),SITE_WALL),np.full(len(fence),FENCE)])
    sampler=lambda xy:np.zeros(len(xy)); walls=site_wall_features(ps,sampler); fences=fence_features(ps,sampler)
    elements,qa=build_site_elements(walls,fences)
    assert not qa and abs(walls[0].thickness-.25)<.03 and abs(fences[0].height-np.percentile(fence[:,2],95))<.05
    entities=to_dxf_entities(elements,LocalFrame((2600000,1200000,500)))
    assert entities["S-WALL"][0]["closed"] and entities["S-FENC"][0]["type"]=="polyline3d"
