"""The 2D review canvas and the IFC viewer JS still parse, and every assistant code path is guarded by its props."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "samcloud" / "app"
CANVAS_JS = APP / "dxf_review_canvas.js"
VIEWER_JS = APP / "ifc_viewer.js"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _block_after(source: str, start: int) -> str:
    """The text of the ``{ ... }`` block that opens at or after ``start`` (brace matching, no strings with braces used)."""
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening:index + 1]
    raise AssertionError("unbalanced braces")


def _methods_section(source: str) -> str:
    return source[source.index("  methods: {"):]


@pytest.mark.parametrize("path", [CANVAS_JS, VIEWER_JS], ids=["canvas", "viewer"])
def test_js_module_syntax(path: Path, tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    target = tmp_path / f"{path.stem}.mjs"
    target.write_text(_source(path), encoding="utf-8")
    subprocess.run([node, "--check", str(target)], check=True, capture_output=True, text=True, timeout=60)


def test_canvas_declares_the_assistant_props() -> None:
    source = _source(CANVAS_JS)
    props = _block_after(source, source.index("  props: {"))
    for name in ("pickEnabled: Boolean", "selectionOverlay: Array", "previewOverlay: Array"):
        assert name in props


def test_canvas_pick_is_only_emitted_when_pick_enabled() -> None:
    source = _source(CANVAS_JS)
    emits = [match.start() for match in re.finditer(r'\$emit\("pick"', source)]
    assert len(emits) == 1
    guard = source.rfind("if (this.pickEnabled", 0, emits[0])
    assert guard != -1 and emits[0] < guard + len(_block_after(source, guard)) + 1
    # click-vs-drag threshold: below 4 px movement and 400 ms
    assert "moved < 4" in source and "< 400" in source


def test_canvas_overlays_are_only_drawn_when_given() -> None:
    source = _source(CANVAS_JS)
    call = source.index("this.drawOverlays(ctx)")
    guard = source.rfind("if (", 0, call)
    condition = source[guard:call]
    assert "this.previewOverlay && this.previewOverlay.length" in condition
    assert "this.selectionOverlay && this.selectionOverlay.length" in condition
    for colour in ("#4bc692", "#ef6b73", "#ffb74d"):  # added green, removed red, moved / modified amber
        assert colour in source
    assert "setLineDash([6, 4])" in source  # "before" rings dashed


def test_canvas_pan_and_measure_paths_unchanged() -> None:
    source = _source(CANVAS_JS)
    on_down = _block_after(source, source.index("    onDown(event) {"))
    assert 'this.$emit("measure"' in on_down and "setPointerCapture" in on_down
    on_move = _block_after(source, source.index("    onMove(event) {"))
    assert "this.view.ox = this.drag.ox + (x - this.drag.x);" in on_move


def test_viewer_declares_the_assistant_props_without_name_clashes() -> None:
    source = _source(VIEWER_JS)
    props = _block_after(source, source.index("  props: {"))
    for name in ("pickEnabled: Boolean", "pickFloor: Number", "selectedMany: Array", "highlight: Object", "boxes: Array"):
        assert name in props
    methods = _methods_section(source)
    declared = set(re.findall(r"^    ([A-Za-z_]\w*)\(", methods, re.MULTILINE))
    prop_names = set(re.findall(r"^    ([A-Za-z_]\w*):", props, re.MULTILINE))
    assert not (declared & prop_names), declared & prop_names  # a method named like a prop breaks the Vue component
    assert "pickHighlight" in declared


def test_viewer_assistant_pick_is_guarded() -> None:
    source = _source(VIEWER_JS)
    pick = _block_after(source, source.index("    pick(event) {"))
    guard = pick.index("if (this.pickEnabled)")
    branch = _block_after(pick, guard)
    assert "ctrl:" in branch and "point:" in branch and "return;" in branch
    after = pick[guard + len(branch):]
    # without pick-enabled the old behaviour: local highlight + pick {global_id, ifc_class}
    assert "this.pickHighlight(chosen, true);" in after
    assert 'this.$emit("pick", { global_id: chosen ? chosen.guid : null, ifc_class: chosen ? chosen.ifcClass : "" });' in after


def test_viewer_marks_and_boxes_are_noops_without_props() -> None:
    source = _source(VIEWER_JS)
    assert "if (this.hasMarks()) this.applyMarks();" in source  # model load / class map only repaint with marks
    apply_boxes = _block_after(source, source.index("    applyBoxes() {"))
    assert "const list = Array.isArray(this.boxes) ? this.boxes : [];" in apply_boxes
    assert "if (this.boxes && this.boxes.length) this.applyBoxes();" in source
    paint = _block_after(source, source.index("    paint(element, on) {"))
    assert "this.markColor(element)" in paint
    mark_color = _block_after(source, source.index("    markColor(element) {"))
    assert "return null;" in mark_color


# ------------------------------------------------------------------------------------------------- behaviour in node

_CANVAS_SCRIPT = r"""
import comp from "%(url)s";
const out = {};
function make(scrollsOnFocus) {
  const emitted = [];
  const vm = {};
  for (const [name, fn] of Object.entries(comp.methods)) vm[name] = fn.bind(vm);
  let top = 300;  // the canvas starts 300 px below the window top and is taller than the window
  const canvas = {
    focusArgs: null, style: {},
    focus(args) { this.focusArgs = args || null; if (scrollsOnFocus || !(args && args.preventScroll)) top -= 300; },
    getBoundingClientRect() { return { left: 40, top, width: 312, height: 1278 }; },
    setPointerCapture() {}, hasPointerCapture() { return true; }, releasePointerCapture() {},
  };
  Object.assign(vm, {
    view: { k: 1, ox: 0, oy: 0, fitted: false }, cssWidth: 312, cssHeight: 1278, pickEnabled: true, tool: "pan",
    measurePoints: [], backgroundBounds: %(bounds)s, imageBounds: null, requestDraw() {},
    $emit(name, args) { emitted.push([name, args]); }, $refs: { canvas, status: { textContent: "" } },
  });
  vm.fit();
  return { vm, canvas, emitted, clientAt(wx, wy) { const [sx, sy] = vm.toScreen(wx, wy); return [40 + sx, top + sy]; } };
}
for (const scrolls of [false, true]) {
  const t = make(scrolls);
  const [cx, cy] = t.clientAt(%(x)s, %(y)s);
  t.vm.onMove({ clientX: cx, clientY: cy });
  const status = t.vm.$refs.status.textContent;
  t.vm.onDown({ button: 0, clientX: cx, clientY: cy, pointerId: 1 });
  t.vm.onUp({ button: 0, clientX: cx, clientY: cy, pointerId: 1, ctrlKey: true });
  out[scrolls ? "scrolling" : "normal"] = { status, emitted: t.emitted, focus: t.canvas.focusArgs };
  // measure tool: the first point is taken before focus() too
  const m = make(scrolls);
  m.vm.tool = "measure";
  const [mx, my] = m.clientAt(%(x)s, %(y)s);
  m.vm.onDown({ button: 0, clientX: mx, clientY: my, pointerId: 1 });
  out[scrolls ? "measure_scrolling" : "measure"] = m.vm.measurePoints[0];
}
console.log(JSON.stringify(out));
"""


def test_canvas_pick_and_hover_invert_the_drawing_transform(tmp_path: Path) -> None:
    """A known label drawn with toScreen() (real-project like background bounds, not starting at 0) is picked back at
    its plan coordinates, also when the browser scrolls the tall canvas on focus (the bug seen on a real project)."""
    import json

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    module = tmp_path / "dxf_review_canvas.mjs"
    module.write_text(_source(CANVAS_JS), encoding="utf-8")
    label = (9.6179, 5.2617)  # e.g. a room label of a plan whose frame origin is [-7, -9, 0]
    script = tmp_path / "pick.mjs"
    script.write_text(_CANVAS_SCRIPT % {"url": module.as_uri(), "bounds": "[0.7461, -0.7980, 15.3861, 11.8820]",
                                        "x": label[0], "y": label[1]}, encoding="utf-8")
    result = json.loads(subprocess.run([node, str(script)], check=True, capture_output=True, text=True, encoding="utf-8", timeout=60).stdout)
    assert result["normal"]["focus"] == {"preventScroll": True}
    assert re.findall(r"-?\d+\.\d+", result["normal"]["status"]) == ["9.618", "5.262"]  # the status bar shows the label position
    for case in ("normal", "scrolling"):
        (name, args), = result[case]["emitted"]
        assert name == "pick" and args["ctrl"] is True
        assert (args["x"], args["y"]) == pytest.approx(label, abs=1e-6)
    for case in ("measure", "measure_scrolling"):
        assert result[case] == pytest.approx(list(label), abs=1e-6)


_VIEWER_SCRIPT = r"""
import comp from "%(url)s";
const vm = {};
for (const [name, fn] of Object.entries(comp.methods)) vm[name] = fn.bind(vm);
function mesh() { return { material: { emissive: { hex: 0, setHex(h) { this.hex = h; } }, emissiveIntensity: 1, userData: { baseEmissive: 0 } } }; }
const meshes = { g1: mesh(), g2: mesh() };
const index = () => Object.keys(meshes).map((guid) => ({ guid, meshes: [meshes[guid]], root: { visible: true } }));
Object.assign(vm, { elements: index(), pickedElement: null, marked: new Set(), selectedManySet: new Set(), requestDraw() {},
                    selectedMany: ["g1", "g2"], highlight: { g2: "#ffb74d" } });
const hex = () => Object.fromEntries(Object.entries(meshes).map(([guid, m]) => [guid, m.material.emissive.hex]));
const out = {};
vm.applyMarks();
out.selected = hex();
vm.elements = index();  // a class-by-node update re-indexes the elements (new objects, same materials)
vm.selectedMany = [];
vm.highlight = {};
vm.applyMarks();
out.cleared = hex();
console.log(JSON.stringify(out));
"""


def test_viewer_marks_clear_after_reindex_and_preview_colour_wins(tmp_path: Path) -> None:
    import json

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    module = tmp_path / "ifc_viewer.mjs"
    module.write_text(_source(VIEWER_JS), encoding="utf-8")
    script = tmp_path / "marks.mjs"
    script.write_text(_VIEWER_SCRIPT % {"url": module.as_uri()}, encoding="utf-8")
    result = json.loads(subprocess.run([node, str(script)], check=True, capture_output=True, text=True, encoding="utf-8", timeout=60).stdout)
    assert result["selected"] == {"g1": 0x2A6BFF, "g2": 0xFFB74D}  # g2 is selected AND in the preview: amber wins
    assert result["cleared"] == {"g1": 0, "g2": 0}  # chip x / clear selection removes the tint after a re-index
