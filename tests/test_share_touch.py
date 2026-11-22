"""Touch navigation of the share viewer: viewport/CSS contract and a node-driven gesture test."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from samcloud.share import export

TEMPLATE = export.TEMPLATE_DIRECTORY

_HARNESS = r"""
// Real Potree View maths (class View of the bundled potree.js) on real three.js vectors.
const fs = require("fs");
const THREE = require(process.argv[3]);
const potree = fs.readFileSync(process.argv[4], "utf8");
const from = potree.indexOf("	class View{"), to = potree.indexOf("	class Scene$1");
const View = new Function("Vector3", "return (" + potree.slice(from, to).trim().replace(/;\s*$/, "") + ")")(THREE.Vector3);
const V3 = THREE.Vector3;
const view = new View();
let zoomed = null, bbox = null, nodes = [];
const cloud = { updateMatrixWorld(){}, matrixWorld: new THREE.Matrix4(),
  get visibleNodes(){ return nodes; }, pcoGeometry: {} };
const viewer = { scene:{view, pointclouds:[cloud], getActiveCamera:()=>({})}, getFOV:()=>60, getBoundingBox:()=>bbox,
  earthControls:{stop(){}, stopTweens(){}, tweens:[], zoomToLocation(m){zoomed=m;}} };
const handlers = {};
const area = { addEventListener(t,f,o){handlers[t]={f,o};}, getBoundingClientRect(){return {left:0,top:0,width:400,height:800};} };
global.window = {}; global.document = {getElementById:()=>({}),title:""}; global.setTimeout = ()=>0;
let pickBehaviour = () => null, pickCalls = 0;
eval(fs.readFileSync(process.argv[2],"utf8"));
// defined only after the viewer script was evaluated (otherwise its start() would try to build a Potree.Viewer)
global.Potree = {Utils: {getMousePointCloudIntersection: (...a) => { pickCalls++; return pickBehaviour(...a); }}};
window.samcloudInstallTouchControls(viewer, area);
let prevented = 0, stopped = 0;
const ev = (type, touches, changed) => ({type, touches, changedTouches: changed||touches,
  preventDefault(){prevented++;}, stopPropagation(){stopped++;}});
const T = (x,y)=>({clientX:x,clientY:y});
const fire = (type, touches, changed) => handlers[type].f(ev(type, touches, changed));
const box = (c, h) => new THREE.Box3(new V3(c.x-h,c.y-h,c.z-h), new V3(c.x+h,c.y+h,c.z+h));
const out = {passive: Object.values(handlers).every(h=>h.o.passive===false && h.o.capture===true),
  types: Object.keys(handlers).sort()};
const front = (pivot) => view.direction.dot(pivot.clone().sub(view.position)) / pivot.distanceTo(view.position);

// Scenario A (as measured in the browser): camera above the house looking down, data near z 0..3 but the loose octree
// cube centre is at z 7.9; no pick hits. Visible nodes describe the real data.
const data = new V3(0, 0, 1.5), cube = new V3(-1.49, -0.08, 7.87);
function setup(withNodes) {
  view.yaw = 0.3; view.pitch = -0.785; view.radius = 1;
  view.position = data.clone().sub(view.direction.clone().multiplyScalar(18)); view.position.z += 2;
  bbox = box(cube, 7.9);
  cloud.pcoGeometry = {};
  nodes = withNodes ? [{geometryNode:{numPoints:5000}, getBoundingBox:()=>box(data, 3)},
                      {geometryNode:{numPoints:3000}, getBoundingBox:()=>box(new V3(1,1,1.2), 2)}] : [];
}
function dragThenPinch(withNodes) {
  setup(withNodes);
  const startZ = view.position.z, startPitch = view.pitch;
  fire("touchstart",[T(200,400)]);
  const pivot = view.getPivot();                       // pivot chosen from the data at touchstart
  const pitchAfterStart = view.pitch;
  for (let i = 1; i <= 4; i++) fire("touchmove",[T(200 + 3*i,400)]);   // purely horizontal, 12 px
  const res = {pitchStart: pitchAfterStart, pitchEnd: view.pitch, startPitch,
    pivotMoved: view.getPivot().distanceTo(pivot), distance: view.position.distanceTo(pivot),
    radius: view.radius, front: front(pivot), above: view.position.z > pivot.z, startZ, z: view.position.z, pivot};
  fire("touchend",[],[T(212,400)]);
  const before = view.position.distanceTo(pivot);
  fire("touchstart",[T(100,300),T(300,300)]);
  const pivot2 = view.getPivot();
  fire("touchmove",[T(60,300),T(340,300)]);
  res.pinchCloser = view.position.distanceTo(pivot2) < before - 0.5;
  res.pinchFront = front(pivot2) > 0.99; res.pinchAbove = view.position.z > pivot2.z;
  res.pivotNearData = pivot.distanceTo(data) < pivot.distanceTo(cube);
  fire("touchend",[],[T(60,300)]);
  return res;
}
out.nodes = dragThenPinch(true); out.cubeOnly = dragThenPinch(false);

// Scenario C (browser log): initial view NOT established - camera at the origin, pitch -0.785, yaw 0.785, radius 8.01;
// every pick fails and the only fallback is the octree cube centre (-1.49, -0.08, 7.87), which lies ABOVE the camera.
function scenarioC() {
  view.position = new V3(0, 0, 0); view.yaw = 0.785; view.pitch = -0.785; view.radius = 8.01;
  bbox = box(cube, 7.9); nodes = [];
  fire("touchstart",[T(200,400)]);
  const afterStart = {pitch: view.pitch, yaw: view.yaw};
  for (let i = 1; i <= 4; i++) fire("touchmove",[T(200 + 3*i,400)]);
  fire("touchend",[],[T(212,400)]);
  fire("touchstart",[T(100,300),T(300,300)]); fire("touchmove",[T(60,300),T(340,300)]); fire("touchend",[],[T(60,300)]);
  return {afterStart, pitch: view.pitch, direction: view.direction.z};
}
out.originStart = scenarioC();
// a pivot above the camera is discarded for the next candidate (a node centre below the camera)
view.position = new V3(0, 0, 20); view.yaw = 0.785; view.pitch = -0.785; view.radius = 1;
bbox = box(cube, 7.9);
nodes = [{geometryNode:{numPoints:100}, getBoundingBox:()=>box(new V3(0,0,25), 1)}];  // above camera: unusable
fire("touchstart",[T(350,700)]);
out.aboveDiscarded = {pitch: view.pitch, pivotZ: view.getPivot().z};
fire("touchend",[],[T(350,700)]);

// Potree 1.8 style nodes: no numPoints on the node itself, getNumPoints()/getBoundingBox(), geometryNode.numPoints = -1
setup(false); view.radius = 1;
nodes = [{getNumPoints:()=>6000, getBoundingBox:()=>box(data, 3), geometryNode:{numPoints:-1}},
         {getNumPoints:()=>2000, getBoundingBox:()=>box(new V3(1,1,1.2), 2), geometryNode:{numPoints:-1}}];
fire("touchstart",[T(350,700)]);
out.potree18Nodes = {pivot: view.getPivot(), nearData: view.getPivot().distanceTo(data) < view.getPivot().distanceTo(cube),
  info: window.samcloudShareTouch ? null : "n/a"};
fire("touchend",[],[T(350,700)]);
out.potree18Nodes.pivot = null;

// generic gestures on a view with a meaningful radius
setup(true); view.radius = view.position.distanceTo(data);
const p1 = view.position.clone();
fire("touchstart",[T(100,300),T(300,300)]); fire("touchmove",[T(70,350),T(270,350)]);
out.panMoved = !view.position.equals(p1);
fire("touchend",[],[T(70,350)]);
setup(true);
fire("touchstart",[T(10,10)]); fire("touchmove",[T(10,5000)]); fire("touchend",[],[T(10,5000)]);
out.pitchClamped = Math.abs(view.pitch) <= Math.PI/2 - 0.05 + 1e-9;
fire("touchstart",[T(200,400)]); fire("touchend",[],[T(200,400)]);
fire("touchstart",[T(202,402)]); fire("touchend",[],[T(202,402)]);
out.doubleTap = zoomed;
// pick throws (pick shader cannot compile): picking is disabled for the session, pivot falls back to the nodes
setup(true); view.radius = 1;
pickBehaviour = () => { throw new Error("could not compile shader pointcloud: GL_EXT_frag_depth"); };
fire("touchstart",[T(300,600)]);
const afterThrow = {pivot: view.getPivot(), disabled: window.samcloudShareFlags.pickingDisabled, calls: pickCalls};
fire("touchend",[],[T(300,600)]);
const callsBefore = pickCalls;
fire("touchstart",[T(250,650)]); fire("touchend",[],[T(250,650)]);
out.pickFailure = {disabled: afterThrow.disabled, nearData: afterThrow.pivot.distanceTo(data) < afterThrow.pivot.distanceTo(cube),
  noMorePicks: pickCalls === callsBefore, firstCalls: afterThrow.calls};
zoomed = null;
fire("touchstart",[T(10,20)]); fire("touchend",[],[T(10,20)]);
fire("touchstart",[T(12,22)]); fire("touchend",[],[T(12,22)]);
out.doubleTapWithoutPicking = zoomed;
out.prevented = prevented > 8 && stopped === prevented;
const plain = r => { const o = {...r}; delete o.pivot; return o; };
out.nodes = plain(out.nodes); out.cubeOnly = plain(out.cubeOnly);
console.log(JSON.stringify(out));
"""


def test_viewer_page_blocks_browser_zoom_and_scroll_gestures() -> None:
    html = (TEMPLATE / "index.html").read_text(encoding="utf-8")
    assert 'name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no"' in html
    css = (TEMPLATE / "css" / "share.css").read_text(encoding="utf-8")
    assert re.search(r"\.share-area[^{]*\{[^}]*touch-action:\s*none", css)
    assert re.search(r"\.share-area \.annotation[^{]*\{[^}]*pointer-events:\s*none", css), "labels never swallow touches"
    assert re.search(r"\.share-legend[^{]*\{[^}]*overscroll-behavior:\s*contain", css)
    js = (TEMPLATE / "js" / "share_viewer.js").read_text(encoding="utf-8")
    assert "passive: false" in js and "capture: true" in js and "zoomToLocation" in js
    assert "if (touched && force !== true) return;" in js, "no automatic fit after the first touch"
    assert "determinePivot" in js and "cloudCentre" in js and "initialFit" in js and "TWEEN.removeAll" in js
    assert "new THREE.Vector2(x - rect.left, y - rect.top)" in js and "defaultOrientation" in js
    # the initial view is the reset routine, called right after addPointCloud (no wait for visible nodes: deadlock)
    assert "boundingBox.isEmpty()" in js and "const initialFit" in js
    then = js[js.index("viewer.scene.addPointCloud(cloud);"):]
    assert then.index("initialFit();") < 400, "fit(true) runs right after addPointCloud"
    assert "waitForCloud" not in js and "visibleNodes.length > 0" not in js
    assert "[500, 1500]" in js and "if (!touched) fit(true)" in js
    assert 'addEventListener("resize"' in js and 'addEventListener("orientationchange"' in js


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_touch_gestures_against_the_real_potree_view_maths(tmp_path: Path) -> None:
    harness = tmp_path / "harness.js"
    harness.write_text(_HARNESS, encoding="utf-8")
    three = export.POTREE_ASSETS / "libs" / "three.js" / "build" / "three.min.js"
    potree = export.POTREE_ASSETS / "build" / "potree" / "potree.js"
    result = subprocess.run(["node", str(harness), str(TEMPLATE / "js" / "share_viewer.js"), str(three), str(potree)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["passive"] is True and out["types"] == ["touchcancel", "touchend", "touchmove", "touchstart"]
    for name in ("nodes", "cubeOnly"):
        run = out[name]
        assert run["pitchEnd"] == pytest.approx(run["pitchStart"], abs=1e-9), "a horizontal drag keeps the pitch"
        assert run["pitchStart"] < 0, "still looking down"
        assert run["above"] and run["z"] > run["startZ"] - 5, "camera stays above the pivot"
        assert run["pivotMoved"] < 1e-6 and run["front"] > 0.999, "direction keeps pointing at the pivot"
        assert run["radius"] == pytest.approx(run["distance"], abs=1e-6)
        assert run["pinchCloser"] and run["pinchFront"] and run["pinchAbove"], "pinch dollies without crossing the pivot"
    assert out["originStart"]["afterStart"]["pitch"] < 0 and out["originStart"]["pitch"] < 0, "no flip to positive pitch"
    assert out["originStart"]["direction"] < 0, "camera keeps looking down"
    assert out["aboveDiscarded"]["pitch"] < 0 and out["aboveDiscarded"]["pivotZ"] < 20
    assert out["potree18Nodes"]["nearData"], "getNumPoints()/getBoundingBox() nodes are weighted correctly"
    assert out["nodes"]["pivotNearData"], "pivot follows the loaded nodes, not the loose octree cube"
    assert out["panMoved"] and out["pitchClamped"]
    assert out["doubleTap"] == {"x": 202, "y": 402}
    assert out["pickFailure"]["disabled"] is True and out["pickFailure"]["nearData"], "pick error: node fallback"
    assert out["pickFailure"]["noMorePicks"] and out["pickFailure"]["firstCalls"] >= 1
    assert out["doubleTapWithoutPicking"] is None, "double tap does not call the picking zoom once picking is disabled"
    assert out["prevented"] is True


_PAN_SCENARIO = r"""
// Two-finger pan must stick: no snap back when fingers are lifted one by one or when the next gesture starts.
const same = (a, b) => a.distanceTo(b) < 1e-9;
const pan = {};
setup(true); view.radius = view.position.distanceTo(data);
const startPosition = view.position.clone(), startYaw = view.yaw, startPitch = view.pitch;
fire("touchstart",[T(100,300),T(300,300)]);
for (let i = 1; i <= 6; i++) fire("touchmove",[T(100 - 25*i, 300 + 12*i),T(300 - 25*i, 300 + 12*i)]);   // 150 px left, 72 px down
const panned = view.position.clone();
pan.moved = panned.distanceTo(startPosition) > 1;
// fingers leave one after the other; the remaining finger wiggles a little
fire("touchend",[T(150,372)],[T(-50,372)]);
fire("touchmove",[T(155,380)]); fire("touchmove",[T(160,390)]);
pan.afterWiggle = view.position.distanceTo(panned);
fire("touchend",[],[T(160,390)]);
pan.afterRelease = view.position.distanceTo(panned);
// frames of the viewer update loop (controls update with their deltas) change nothing either
for (let i = 0; i < 5; i++) { viewer.earthControls.stop(); }
// the next gesture starts: the camera must neither move nor turn
fire("touchstart",[T(200,400)]);
pan.afterNextStart = view.position.distanceTo(panned);
pan.yawChange = Math.abs(view.yaw - startYaw) + Math.abs(view.pitch - startPitch);
const pivotBefore = view.getPivot();
for (let i = 1; i <= 8; i++) fire("touchmove",[T(200 + 5*i, 400)]);
const pivotAfter = view.getPivot();
pan.orbitPivotDrift = pivotAfter.distanceTo(pivotBefore);
pan.orbitRadius = Math.abs(view.position.distanceTo(pivotAfter) - view.radius);
pan.yawTurned = Math.abs(view.yaw - startYaw) > 0.1;
pan.pivotMovedWithPan = pivotBefore.distanceTo(data) > 1;
fire("touchend",[],[T(240,400)]);
// pinch combined with pan, finger lift order reversed: still sticks
const beforePinch = view.position.clone();
fire("touchstart",[T(100,300),T(300,300)]);
for (let i = 1; i <= 4; i++) fire("touchmove",[T(100 - 10*i, 300),T(300 + 10*i, 300 - 5*i)]);
const pinched = view.position.clone();
pan.pinchMoved = pinched.distanceTo(beforePinch) > 0.1;
fire("touchend",[T(140,300)],[T(340,280)]);
fire("touchmove",[T(150,305)]);
fire("touchend",[],[T(150,305)]);
pan.pinchSticks = view.position.distanceTo(pinched);
console.log(JSON.stringify(pan));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_two_finger_pan_sticks_and_next_orbit_uses_the_moved_pivot(tmp_path: Path) -> None:
    """Old logic: state.pivot stayed at the pre-pan point, so the leftover finger (2 -> 1 touches) orbited around it and
    pulled the camera back; the next touchstart also lookAt()-turned the camera to the data centre (view snapped back)."""
    harness = tmp_path / "harness_pan.js"
    harness.write_text(_HARNESS[:_HARNESS.index("function dragThenPinch")] + _PAN_SCENARIO, encoding="utf-8")
    three = export.POTREE_ASSETS / "libs" / "three.js" / "build" / "three.min.js"
    potree = export.POTREE_ASSETS / "build" / "potree" / "potree.js"
    result = subprocess.run(["node", str(harness), str(TEMPLATE / "js" / "share_viewer.js"), str(three), str(potree)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    pan = json.loads(result.stdout.strip().splitlines()[-1])
    assert pan["moved"], "the two-finger pan moves the camera"
    assert pan["afterWiggle"] < 1e-9, "the leftover finger after lifting one finger does not orbit around a stale pivot"
    assert pan["afterRelease"] < 1e-9
    assert pan["afterNextStart"] < 1e-9 and pan["yawChange"] < 1e-9, "next gesture start neither moves nor turns the camera"
    assert pan["pivotMovedWithPan"] and pan["orbitPivotDrift"] < 1e-6, "one-finger orbit pivots around the moved pivot"
    assert pan["orbitRadius"] < 1e-6 and pan["yawTurned"] is True, "the orbit really rotated"
    assert pan["pinchMoved"] and pan["pinchSticks"] < 1e-9, "pinch + pan also sticks"
