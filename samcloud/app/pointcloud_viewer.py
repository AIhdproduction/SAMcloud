"""Reusable lightweight 3D point cloud viewer for local project products."""

from __future__ import annotations

import json
from html import escape

from nicegui import ui


DEFAULT_CLASS_COLORS = (
    (189, 189, 189),  # Unknown / noise
    (166, 124, 82),   # Ground
    (77, 77, 77),     # Road
    (184, 184, 184),  # Sidewalk
    (255, 210, 63),   # Curb
    (126, 200, 80),   # Low vegetation
    (34, 139, 34),    # Trees
    (217, 83, 79),    # Buildings
    (0, 162, 232),    # Vehicles
    (255, 140, 0),    # Construction machinery
    (139, 90, 43),    # Poles
    (255, 0, 255),    # Traffic signs
    (138, 43, 226),   # Fences
    (0, 206, 209),    # Wires
    (255, 215, 0),    # Scaffolding
    (255, 99, 71),    # Containers
    (30, 144, 255),   # Water
)


_STYLE = """
<style>
.sam-cloud-viewer { position:relative; width:100%; height:calc(100vh - 150px); min-height:480px; overflow:hidden; border:1px solid #28475b; border-radius:9px; background:#050b12; }
.sam-cloud-viewer:fullscreen { width:100vw; height:100vh; min-height:100vh; border:0; border-radius:0; }
.sam-cloud-viewer canvas { display:block; width:100%; height:100%; cursor:grab; touch-action:none; }
.sam-cloud-viewer canvas.is-dragging { cursor:grabbing; }
.sam-cloud-toolbar { position:absolute; top:12px; left:12px; right:12px; display:flex; flex-wrap:wrap; gap:8px; align-items:center; pointer-events:none; }
.sam-cloud-toolbar > * { pointer-events:auto; }
.sam-cloud-toolbar button { border:1px solid #47718a; border-radius:5px; padding:6px 10px; background:#102131; color:#dcecf6; cursor:pointer; }
.sam-cloud-toolbar button.is-selected { border-color:#26b7ef; background:#16516c; }
.sam-cloud-toolbar label { display:flex; align-items:center; gap:6px; padding:5px 9px; border:1px solid #28475b; border-radius:5px; background:rgba(9,19,29,.86); color:#b9ccda; font-size:12px; }
.sam-cloud-status { position:absolute; left:12px; bottom:12px; padding:6px 9px; border-radius:5px; background:rgba(9,19,29,.86); color:#b9ccda; font-size:12px; }
.sam-cloud-empty { display:grid; place-items:center; height:100%; padding:30px; color:#96aabe; text-align:center; }
.sam-cloud-class-panel { display:none; position:absolute; z-index:2; top:64px; right:12px; bottom:12px; width:min(340px,34vw); overflow:auto; padding:12px; border:1px solid #365c74; border-radius:8px; background:rgba(9,19,29,.94); color:#eaf2f8; }
.sam-cloud-viewer:fullscreen .sam-cloud-class-panel { display:flex; flex-direction:column; gap:8px; }
.sam-cloud-class-panel h3 { margin:0 0 4px; font-size:15px; }.sam-cloud-class-actions { display:flex; gap:6px; flex-wrap:wrap; }.sam-cloud-class-actions button,.sam-cloud-class-row button { border:1px solid #47718a; border-radius:5px; padding:5px 8px; background:#102131; color:#dcecf6; cursor:pointer; font-size:12px; }.sam-cloud-class-row { display:flex; align-items:center; gap:7px; padding:4px 0; border-bottom:1px solid rgba(113,154,181,.16); font-size:12px; }.sam-cloud-class-row label { flex:1; display:flex; align-items:center; gap:7px; cursor:pointer; }.sam-cloud-class-row input { accent-color:#26b7ef; }
.sam-cloud-camera-card { display:none; position:absolute; z-index:3; right:12px; bottom:12px; width:min(320px,42vw); padding:10px; border:1px solid #47718a; border-radius:7px; background:rgba(9,19,29,.94); color:#eaf2f8; font-size:12px; }.sam-cloud-camera-card.is-open { display:block; }.sam-cloud-camera-card img { display:block; width:100%; max-height:210px; margin-top:8px; object-fit:contain; background:#050b12; }.sam-cloud-camera-card a { display:inline-block; margin-top:7px; color:#59c8f3; }
</style>
"""


def render_pointcloud_viewer(
    kind: str,
    *,
    title: str,
    class_colors: list[list[int]] | None = None,
    class_labels: list[str] | None = None,
    marker_endpoint: str | None = None,
) -> None:
    """Render a canvas viewer backed by the active project's compact point preview."""
    from samcloud.viewer.potree import potree_available

    if potree_available():
        from samcloud.app.potree_viewer import render_potree_viewer

        render_potree_viewer(
            kind, title=title, class_colors=class_colors or default_class_colors(),
            class_labels=class_labels, marker_endpoint=marker_endpoint,
            camera_endpoint="/api/cameras/sparse" if kind == "sparse" else None,
        )
        return

    element_id = f"sam-cloud-{kind}"
    endpoint = f"/api/pointcloud/{kind}"
    class_panel = _class_panel_html(class_labels or [])
    camera_controls = (
        '<button type="button" data-action="cameras" class="is-selected">Kamerapositionen</button>'
        '<button type="button" data-action="camera-path">Kamerapfad</button>'
        '<button type="button" data-action="camera-photos">Foto-Ansichten</button>'
        '<button type="button" data-action="camera-colour">Reprojektionsfehler</button>'
        if kind == "sparse" else ""
    )
    camera_sliders = (
        '<label>Kameragrösse <input data-action="camera-size" type="range" min="0.25" max="8" step="0.25" value="1"></label>'
        '<label>Pfad-Linienbreite <input data-action="camera-path-width" type="range" min="0.5" max="8" step="0.5" value="1.5"></label>'
        '<label>Foto-Grösse <input data-action="camera-photo-size" type="range" min="0.25" max="5" step="0.25" value="1"></label>'
        if kind == "sparse" else ""
    )
    camera_card = '<aside class="sam-cloud-camera-card"></aside>' if kind == "sparse" else ""
    ui.add_head_html(_STYLE)
    ui.html(
        f"""
        <div id="{element_id}" class="sam-cloud-viewer">
          <canvas aria-label="{title}"></canvas>
          <div class="sam-cloud-toolbar">
            <button type="button" data-action="reset">Reset view</button>
            <button type="button" data-action="zoom-out">−</button>
            <button type="button" data-action="zoom-in">+</button>
            <button type="button" data-action="fit">Fit all</button>
            {camera_controls}
            <button type="button" data-action="fullscreen">Fullscreen</button>
            <button type="button" data-view="terrain" class="is-selected">Terrain view</button>
            <button type="button" data-view="orbit">Orbit view</button>
             <button type="button" data-mode="rgb" class="is-selected">RGB</button>
             <button type="button" data-mode="class">Classification</button>
             <label>Point size <input data-action="size" type="range" min="1" max="6" step="0.5" value="1"></label>
             {camera_sliders}
          </div>
          {class_panel}
          {camera_card}
          <div class="sam-cloud-status">Loading point cloud preview...</div>
        </div>
        """,
        sanitize=False,
    )
    ui.add_body_html(
        _script(element_id, endpoint, class_colors or default_class_colors(), marker_endpoint=marker_endpoint,
                camera_endpoint="/api/cameras/sparse" if kind == "sparse" else None)
    )


def default_class_colors(count: int = len(DEFAULT_CLASS_COLORS)) -> list[list[int]]:
    """Return the editable Cyclone-style classification palette."""
    return [list(DEFAULT_CLASS_COLORS[index % len(DEFAULT_CLASS_COLORS)]) for index in range(count)]


def _class_panel_html(class_labels: list[str]) -> str:
    """Build fullscreen-only dense class controls without color editors."""
    if not class_labels:
        return ""
    rows = "".join(
        "<div class=\"sam-cloud-class-row\">"
        f"<label><input type=\"checkbox\" data-class-visible=\"{index}\" checked>{escape(label)}</label>"
        f"<button type=\"button\" data-class-isolate=\"{index}\">Isolate</button>"
        "</div>"
        for index, label in enumerate(class_labels)
    )
    return (
        "<aside class=\"sam-cloud-class-panel\">"
        "<h3>Class visibility</h3>"
        "<div class=\"sam-cloud-class-actions\">"
        "<button type=\"button\" data-class-show-all>Show all</button>"
        "<button type=\"button\" data-class-hide-all>Hide all</button>"
        "</div>"
        f"{rows}</aside>"
    )


def _script(
    element_id: str, endpoint: str, class_colors: list[list[int]] | None = None, *, marker_endpoint: str | None = None,
    camera_endpoint: str | None = None,
) -> str:
    config = json.dumps(
        {
            "element_id": element_id,
            "endpoint": endpoint,
            "class_colors": class_colors or default_class_colors(),
            "marker_endpoint": marker_endpoint,
            "camera_endpoint": camera_endpoint,
        }
    )
    return f"""
<script>
(() => {{
  const config = {config};
  const init = () => {{
    const root = document.getElementById(config.element_id);
    if (!root || root.dataset.ready === '1') return;
    root.dataset.ready = '1';
    const canvas = root.querySelector('canvas');
    const context = canvas.getContext('2d');
    const status = root.querySelector('.sam-cloud-status');
    const cameraSettingsKey = 'samcloud:sparse-camera-settings:v1';
    const savedCameraSettings = (() => {{ try {{ return JSON.parse(localStorage.getItem(cameraSettingsKey) || '{{}}'); }} catch (_error) {{ return {{}}; }} }})();
    const state = {{ points: null, colors: null, classes: null, count: 0, center: [0, 0, 0], baseCenter: [0, 0, 0], span3D: 1, yaw: .65, pitch: -.45, terrainTilt: 0, zoom: 1, viewMode: 'terrain', panX: 0, panY: 0, mode: 'rgb', pointSize: 1, classColors: config.class_colors, classVisible: config.class_colors.map(() => true), markers: [], selectedMarker: '', cameras: [], camerasVisible: savedCameraSettings.positions !== false, cameraPathVisible: savedCameraSettings.path === true, cameraPhotosVisible: savedCameraSettings.photos === true, cameraSize: Number(savedCameraSettings.size) || 1, cameraPathWidth: Number(savedCameraSettings.pathWidth) || 1.5, cameraPhotoSize: Number(savedCameraSettings.photoSize) || 1, cameraImages: new Map(), cameraPhotoLoading: new Set(), cameraErrorColours: false, selectedCamera: '', dragging: false, dragMode: '', lastX: 0, lastY: 0 }};
    const cameraCard = root.querySelector('.sam-cloud-camera-card');
    const classPanel = root.querySelector('.sam-cloud-class-panel');
    const saveCameraSettings = () => {{
      try {{ localStorage.setItem(cameraSettingsKey, JSON.stringify({{ positions: state.camerasVisible, path: state.cameraPathVisible, photos: state.cameraPhotosVisible, size: state.cameraSize, pathWidth: state.cameraPathWidth, photoSize: state.cameraPhotoSize }})); }} catch (_error) {{}}
    }};
    const syncCameraControls = () => {{
      [['cameras', state.camerasVisible], ['camera-path', state.cameraPathVisible], ['camera-photos', state.cameraPhotosVisible]].forEach(([action, selected]) => {{ const control = root.querySelector(`[data-action="${{action}}"]`); if (control) control.classList.toggle('is-selected', selected); }});
      [['camera-size', state.cameraSize], ['camera-path-width', state.cameraPathWidth], ['camera-photo-size', state.cameraPhotoSize]].forEach(([action, value]) => {{ const control = root.querySelector(`[data-action="${{action}}"]`); if (control) control.value = value; }});
    }};
    const syncClassControls = () => {{
      if (!classPanel) return;
      classPanel.querySelectorAll('[data-class-visible]').forEach(control => {{
        control.checked = state.classVisible[Number(control.dataset.classVisible)] !== false;
      }});
    }};
    const setClassVisibility = visibility => {{
      state.classVisible = visibility;
      syncClassControls();
      draw();
    }};
    window.samcloudPointCloudViewers = window.samcloudPointCloudViewers || {{}};
    window.samcloudPointCloudViewers[config.element_id] = {{
      setClassColors: colors => {{ state.classColors = colors; draw(); }},
      setClassVisibility,
    }};
    if (classPanel) {{
      classPanel.querySelectorAll('[data-class-visible]').forEach(control => {{
        control.onchange = () => {{
          state.classVisible[Number(control.dataset.classVisible)] = control.checked;
          draw();
        }};
      }});
      classPanel.querySelectorAll('[data-class-isolate]').forEach(button => {{
        button.onclick = () => {{
          const selected = Number(button.dataset.classIsolate);
          setClassVisibility(state.classVisible.map((_, index) => index === selected));
        }};
      }});
      classPanel.querySelector('[data-class-show-all]').onclick = () => {{
        setClassVisibility(state.classVisible.map(() => true));
      }};
      classPanel.querySelector('[data-class-hide-all]').onclick = () => {{
        setClassVisibility(state.classVisible.map(() => false));
      }};
    }}

    const resize = () => {{
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * ratio));
      canvas.height = Math.max(1, Math.floor(rect.height * ratio));
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      if (state.points) {{ state.zoom = 1; state.panX = 0; state.panY = 0; state.center = [...state.baseCenter]; }}
      draw();
    }};
    const draw = () => {{
      const rect = canvas.getBoundingClientRect();
      const width = rect.width, height = rect.height;
      context.clearRect(0, 0, width, height);
      context.fillStyle = '#050b12'; context.fillRect(0, 0, width, height);
      if (!state.points) return;
      const fitSpan = state.span3D;
      const scale = Math.min(width, height) * .82 * state.zoom / fitSpan;
      const size = state.pointSize;
      const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw);
      const cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
      const ct = Math.cos(state.terrainTilt), st = Math.sin(state.terrainTilt);
      for (let i = 0; i < state.count; i++) {{
        if (state.classVisible[state.classes[i]] === false) continue;
        const p = i * 3;
        const x = state.points[p] - state.center[0], y = state.points[p + 1] - state.center[1], z = state.points[p + 2] - state.center[2];
        let x1, y1, depth;
        if (state.viewMode === 'terrain') {{
          const groundY = x * sy + y * cy;
          x1 = x * cy - y * sy;
          y1 = groundY * ct + z * st;
          depth = z * ct - groundY * st;
        }} else {{
          const z1 = x * sy + z * cy;
          x1 = x * cy - z * sy;
          y1 = y * cp - z1 * sp;
          depth = y * sp + z1 * cp;
        }}
        const depthFactor = Math.max(.72, Math.min(1.35, 1.0 - depth / (fitSpan * 7)));
        const projectedScale = scale * depthFactor;
        const px = width * .5 + state.panX + x1 * projectedScale, py = height * .5 + state.panY - y1 * projectedScale;
        if (px < -size || py < -size || px > width + size || py > height + size) continue;
        let red, green, blue;
        if (state.mode === 'class') {{
          const color = state.classColors[state.classes[i] % state.classColors.length]; [red, green, blue] = color;
        }} else {{
          const c = i * 3; red = state.colors[c]; green = state.colors[c + 1]; blue = state.colors[c + 2];
        }}
        context.fillStyle = `rgb(${{red}},${{green}},${{blue}})`;
        const point = Math.max(.7, size * Math.min(2.5, Math.max(.8, projectedScale / scale * 1.8)));
        context.fillRect(px, py, point, point);
      }}
      const projectPosition = coordinates => {{
        const x = coordinates[0] - state.center[0], y = coordinates[1] - state.center[1], z = coordinates[2] - state.center[2];
        let x1, y1, depth;
        if (state.viewMode === 'terrain') {{ const groundY = x * sy + y * cy; x1 = x * cy - y * sy; y1 = groundY * ct + z * st; depth = z * ct - groundY * st; }}
        else {{ const z1 = x * sy + z * cy; x1 = x * cy - z * sy; y1 = y * cp - z1 * sp; depth = y * sp + z1 * cp; }}
        const depthFactor = Math.max(.72, Math.min(1.35, 1.0 - depth / (fitSpan * 7)));
        return [width * .5 + state.panX + x1 * scale * depthFactor, height * .5 + state.panY - y1 * scale * depthFactor];
      }};
      const cameraColour = camera => {{
        if (state.cameraErrorColours && Number.isFinite(camera.mean_reprojection_error)) {{
          const error = Math.min(4, Math.max(0, Number(camera.mean_reprojection_error)));
          return `hsl(${{(1 - error / 4) * 122}}, 85%, 55%)`;
        }}
        return `hsl(${{(Number(camera.index || 0) / Math.max(1, state.cameras.length - 1)) * 260}}, 85%, 55%)`;
      }};
      if (state.cameraPathVisible && state.cameras.length > 1) {{
        context.save(); context.strokeStyle = '#6fc9ec'; context.globalAlpha = .75; context.lineWidth = state.cameraPathWidth; context.beginPath();
        state.cameras.forEach((camera, index) => {{ const [px, py] = projectPosition(camera.position); if (index) context.lineTo(px, py); else context.moveTo(px, py); }});
        context.stroke(); context.restore();
      }}
      if (state.camerasVisible && state.cameraPhotosVisible) {{
        const candidates = state.cameras.map(camera => {{
          if (!Array.isArray(camera.position) || camera.position.length !== 3) return null;
          const [px, py] = projectPosition(camera.position);
          return {{ camera, px, py, distance: Math.hypot(px - width * .5, py - height * .5) }};
        }}).filter(item => item && item.px > -120 && item.py > -120 && item.px < width + 120 && item.py < height + 120).sort((a, b) => a.distance - b.distance).slice(0, 72);
        const photoWidth = Math.max(18, Math.min(180, state.cameraPhotoSize * state.span3D * scale * .035));
        for (const item of candidates) {{
          const image = state.cameraImages.get(item.camera.id);
          if (!image && !state.cameraPhotoLoading.has(item.camera.id)) {{
            state.cameraPhotoLoading.add(item.camera.id); const next = new Image(); next.decoding = 'async'; next.onload = () => {{ state.cameraImages.set(item.camera.id, next); state.cameraPhotoLoading.delete(item.camera.id); draw(); }}; next.onerror = () => state.cameraPhotoLoading.delete(item.camera.id); next.src = `/api/camera-thumbnail/${{encodeURIComponent(item.camera.id)}}?size=160`;
          }}
          context.save(); context.globalAlpha = .94; context.fillStyle = '#102131'; context.strokeStyle = '#6fc9ec'; context.lineWidth = 1; context.fillRect(item.px - photoWidth / 2 - 2, item.py - photoWidth * .75 - 2, photoWidth + 4, photoWidth * .75 + 4); context.strokeRect(item.px - photoWidth / 2 - 2, item.py - photoWidth * .75 - 2, photoWidth + 4, photoWidth * .75 + 4);
          if (image) {{ const ratio = image.naturalWidth / Math.max(1, image.naturalHeight); const photoHeight = Math.min(photoWidth * .75, photoWidth / Math.max(.2, ratio)); context.drawImage(image, item.px - photoWidth / 2, item.py - photoHeight - 2, photoWidth, photoHeight); }} context.restore();
        }}
      }}
      if (state.camerasVisible) for (const camera of state.cameras) {{
        if (!Array.isArray(camera.position) || camera.position.length !== 3) continue;
        const [px, py] = projectPosition(camera.position);
        const direction = Array.isArray(camera.direction) ? camera.direction : [0, 0, 1];
        const [dx, dy] = projectPosition([camera.position[0] + direction[0] * state.span3D * .02 * state.cameraSize, camera.position[1] + direction[1] * state.span3D * .02 * state.cameraSize, camera.position[2] + direction[2] * state.span3D * .02 * state.cameraSize]);
        const angle = Math.atan2(dy - py, dx - px); context.save(); context.translate(px, py); context.rotate(angle + Math.PI / 2); context.fillStyle = cameraColour(camera);
        const cameraPixelSize = Math.max(3, state.cameraSize * state.span3D * scale * .008); context.beginPath(); context.moveTo(0, -cameraPixelSize); context.lineTo(cameraPixelSize * .72, cameraPixelSize * .72); context.lineTo(-cameraPixelSize * .72, cameraPixelSize * .72); context.closePath(); context.fill();
        if (camera.id === state.selectedCamera) {{ context.strokeStyle = '#ffffff'; context.lineWidth = 1.5; context.stroke(); }} context.restore();
      }}
      for (const marker of state.markers) {{
        if (!Array.isArray(marker.coordinates) || marker.coordinates.length !== 3) continue;
        const x = marker.coordinates[0] - state.center[0], y = marker.coordinates[1] - state.center[1], z = marker.coordinates[2] - state.center[2];
        let x1, y1, depth;
        if (state.viewMode === 'terrain') {{
          const groundY = x * sy + y * cy; x1 = x * cy - y * sy; y1 = groundY * ct + z * st; depth = z * ct - groundY * st;
        }} else {{
          const z1 = x * sy + z * cy; x1 = x * cy - z * sy; y1 = y * cp - z1 * sp; depth = y * sp + z1 * cp;
        }}
        const depthFactor = Math.max(.72, Math.min(1.35, 1.0 - depth / (fitSpan * 7)));
        const px = width * .5 + state.panX + x1 * scale * depthFactor, py = height * .5 + state.panY - y1 * scale * depthFactor;
        const colors = {{ pass: '#26b7ef', control: '#ffb020', disabled: '#6b7c8a', needs_review: '#ff5a5a' }};
        context.save(); context.fillStyle = colors[marker.state_color] || colors.needs_review;
        context.beginPath(); context.moveTo(px, py - 12); context.lineTo(px + 7, py - 2); context.lineTo(px + 2, py - 2); context.lineTo(px + 2, py + 7); context.lineTo(px - 2, py + 7); context.lineTo(px - 2, py - 2); context.lineTo(px - 7, py - 2); context.closePath(); context.fill();
        if (marker.name === state.selectedMarker) {{ context.strokeStyle = '#dcecf6'; context.lineWidth = 1.5; context.stroke(); }}
        context.font = '12px sans-serif'; context.fillStyle = '#dcecf6'; context.fillText(marker.name, px + 9, py - 7); context.restore();
      }}
    }};
    const fitAll = () => {{ state.zoom = 1; state.panX = 0; state.panY = 0; state.center = [...state.baseCenter]; draw(); }};
    const load = async () => {{
      try {{
        const response = await fetch(config.endpoint, {{ cache: 'no-store' }});
        if (!response.ok) throw new Error((await response.json()).error || 'Point cloud is not available');
        const buffer = await response.arrayBuffer();
        const view = new DataView(buffer);
        if (view.getUint8(0) !== 83 || view.getUint8(1) !== 65 || view.getUint8(2) !== 77 || view.getUint8(3) !== 86) throw new Error('Invalid point cloud preview');
        const count = view.getUint32(8, true), stride = 16, points = new Float32Array(count * 3), colors = new Uint8Array(count * 3), classes = new Uint8Array(count);
        let minX = Infinity, minY = Infinity, minZ = Infinity, maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
        for (let i = 0, offset = 16; i < count; i++, offset += stride) {{
          const x = view.getFloat32(offset, true), y = view.getFloat32(offset + 4, true), z = view.getFloat32(offset + 8, true);
          points[i * 3] = x; points[i * 3 + 1] = y; points[i * 3 + 2] = z;
          colors[i * 3] = view.getUint8(offset + 12); colors[i * 3 + 1] = view.getUint8(offset + 13); colors[i * 3 + 2] = view.getUint8(offset + 14); classes[i] = view.getUint8(offset + 15);
          minX = Math.min(minX, x); minY = Math.min(minY, y); minZ = Math.min(minZ, z); maxX = Math.max(maxX, x); maxY = Math.max(maxY, y); maxZ = Math.max(maxZ, z);
        }}
        state.points = points; state.colors = colors; state.classes = classes; state.count = count;
        const spanX = maxX - minX, spanY = maxY - minY, spanZ = maxZ - minZ;
        state.baseCenter = [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2];
        state.center = [...state.baseCenter];
        state.span3D = Math.max(Math.hypot(spanX, spanY, spanZ), .001);
        status.textContent = `Preview: ${{count.toLocaleString()}} points. Left drag rotates around the nearest point, right drag moves, wheel zooms.`;
        await loadMarkers(); await loadCameras(); resize();
      }} catch (error) {{ status.textContent = `Point cloud unavailable: ${{error.message}}`; }}
    }};
    const loadMarkers = async () => {{
      if (!config.marker_endpoint) return;
      try {{ const response = await fetch(config.marker_endpoint, {{ cache: 'no-store' }}); if (!response.ok) return; const value = await response.json(); state.markers = value.markers || []; draw(); }} catch (_error) {{ }}
    }};
    const loadCameras = async () => {{
      if (!config.camera_endpoint) return;
      try {{ const response = await fetch(config.camera_endpoint, {{ cache: 'no-store' }}); if (!response.ok) return; const value = await response.json(); state.cameras = value.cameras || []; const path = root.querySelector('[data-action="camera-path"]'); if (path && !value.path_available) path.style.display = 'none'; draw(); }} catch (_error) {{ }}
    }};
    const showCamera = camera => {{
      if (!cameraCard || !state.cameraPhotosVisible) return;
      cameraCard.innerHTML = ''; const title = document.createElement('strong'); title.textContent = camera.source_name || camera.image_name || 'Kamera'; cameraCard.append(title);
      const details = document.createElement('div'); const error = Number.isFinite(camera.mean_reprojection_error) ? `${{Number(camera.mean_reprojection_error).toFixed(2)}} px` : '–'; details.textContent = `Aufnahme ${{Number(camera.index) + 1}} · ${{camera.points3d || 0}} 3D-Punkte · Reprojektionsfehler ${{error}}`; cameraCard.append(details);
      const imageUrl = `/api/camera-thumbnail/${{encodeURIComponent(camera.id)}}?size=320`; const image = document.createElement('img'); image.src = imageUrl; image.alt = camera.source_name || camera.image_name || 'Kamerafoto'; cameraCard.append(image);
      const link = document.createElement('a'); link.href = imageUrl; link.target = '_blank'; link.rel = 'noopener'; link.textContent = 'Foto anzeigen'; cameraCard.append(link); cameraCard.classList.add('is-open');
    }};
    const selectCamera = event => {{
      if (!state.camerasVisible || !state.cameraPhotosVisible || !state.cameras.length || !state.points) return false;
      const rect = canvas.getBoundingClientRect(), targetX = event.clientX - rect.left, targetY = event.clientY - rect.top;
      const width = rect.width, height = rect.height, scale = Math.min(width, height) * .82 * state.zoom / state.span3D;
      const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw), cp = Math.cos(state.pitch), sp = Math.sin(state.pitch), ct = Math.cos(state.terrainTilt), st = Math.sin(state.terrainTilt);
      let chosen = null, distance = 12;
      for (const camera of state.cameras) {{ const position = camera.position; if (!Array.isArray(position)) continue; const x = position[0] - state.center[0], y = position[1] - state.center[1], z = position[2] - state.center[2]; let x1, y1, depth; if (state.viewMode === 'terrain') {{ const groundY = x * sy + y * cy; x1 = x * cy - y * sy; y1 = groundY * ct + z * st; depth = z * ct - groundY * st; }} else {{ const z1 = x * sy + z * cy; x1 = x * cy - z * sy; y1 = y * cp - z1 * sp; depth = y * sp + z1 * cp; }} const depthFactor = Math.max(.72, Math.min(1.35, 1.0 - depth / (state.span3D * 7))); const px = width * .5 + state.panX + x1 * scale * depthFactor, py = height * .5 + state.panY - y1 * scale * depthFactor; const current = Math.hypot(px - targetX, py - targetY); if (current < distance) {{ distance = current; chosen = camera; }} }}
      if (!chosen) return false; state.selectedCamera = chosen.id; showCamera(chosen); draw(); return true;
    }};
    const selectMarker = event => {{
      if (!state.markers.length || !state.points) return false;
      const rect = canvas.getBoundingClientRect(), targetX = event.clientX - rect.left, targetY = event.clientY - rect.top;
      const width = rect.width, height = rect.height, scale = Math.min(width, height) * .82 * state.zoom / state.span3D;
      const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw), cp = Math.cos(state.pitch), sp = Math.sin(state.pitch), ct = Math.cos(state.terrainTilt), st = Math.sin(state.terrainTilt);
      let chosen = null, distance = 11;
      for (const marker of state.markers) {{
        if (!Array.isArray(marker.coordinates)) continue;
        const x = marker.coordinates[0] - state.center[0], y = marker.coordinates[1] - state.center[1], z = marker.coordinates[2] - state.center[2]; let x1, y1, depth;
        if (state.viewMode === 'terrain') {{ const groundY = x * sy + y * cy; x1 = x * cy - y * sy; y1 = groundY * ct + z * st; depth = z * ct - groundY * st; }} else {{ const z1 = x * sy + z * cy; x1 = x * cy - z * sy; y1 = y * cp - z1 * sp; depth = y * sp + z1 * cp; }}
        const depthFactor = Math.max(.72, Math.min(1.35, 1.0 - depth / (state.span3D * 7))); const px = width * .5 + state.panX + x1 * scale * depthFactor, py = height * .5 + state.panY - y1 * scale * depthFactor; const current = Math.hypot(px - targetX, py - targetY);
        if (current < distance) {{ distance = current; chosen = marker; }}
      }}
      if (!chosen) return false; state.selectedMarker = chosen.name; window.dispatchEvent(new CustomEvent('samcloud-marker-selected', {{ detail: chosen }})); draw(); return true;
    }};
    const setRotationPivot = (clientX, clientY) => {{
      if (!state.points) return;
      const rect = canvas.getBoundingClientRect(), width = rect.width, height = rect.height;
      const targetX = clientX - rect.left, targetY = clientY - rect.top;
      const fitSpan = state.span3D, scale = Math.min(width, height) * .82 * state.zoom / fitSpan;
      const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw), cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
      const ct = Math.cos(state.terrainTilt), st = Math.sin(state.terrainTilt);
      let nearest = -1, nearestDistance = Infinity;
      for (let i = 0; i < state.count; i++) {{
        const p = i * 3;
        const x = state.points[p] - state.center[0], y = state.points[p + 1] - state.center[1], z = state.points[p + 2] - state.center[2];
        let x1, y1, depth;
        if (state.viewMode === 'terrain') {{
          const groundY = x * sy + y * cy;
          x1 = x * cy - y * sy;
          y1 = groundY * ct + z * st;
          depth = z * ct - groundY * st;
        }} else {{
          const z1 = x * sy + z * cy;
          x1 = x * cy - z * sy;
          y1 = y * cp - z1 * sp;
          depth = y * sp + z1 * cp;
        }}
        const depthFactor = Math.max(.72, Math.min(1.35, 1.0 - depth / (fitSpan * 7)));
        const px = width * .5 + state.panX + x1 * scale * depthFactor;
        const py = height * .5 + state.panY - y1 * scale * depthFactor;
        const distance = (px - targetX) ** 2 + (py - targetY) ** 2;
        if (distance < nearestDistance) {{ nearestDistance = distance; nearest = i; }}
      }}
      if (nearest < 0) return;
      const point = nearest * 3;
      state.center = [state.points[point], state.points[point + 1], state.points[point + 2]];
      state.panX = targetX - width * .5;
      state.panY = targetY - height * .5;
    }};
    root.querySelector('[data-action="reset"]').onclick = () => {{
      if (state.viewMode === 'terrain') {{ state.yaw = 0; state.terrainTilt = 0; }}
      else {{ state.yaw = .65; state.pitch = -.45; }}
      fitAll();
    }};
    root.querySelector('[data-action="fit"]').onclick = fitAll;
    const camerasButton = root.querySelector('[data-action="cameras"]');
    if (camerasButton) camerasButton.onclick = () => {{ state.camerasVisible = !state.camerasVisible; saveCameraSettings(); syncCameraControls(); draw(); }};
    const cameraPathButton = root.querySelector('[data-action="camera-path"]');
    if (cameraPathButton) cameraPathButton.onclick = () => {{ state.cameraPathVisible = !state.cameraPathVisible; saveCameraSettings(); syncCameraControls(); draw(); }};
    const cameraPhotosButton = root.querySelector('[data-action="camera-photos"]');
    if (cameraPhotosButton) cameraPhotosButton.onclick = () => {{ state.cameraPhotosVisible = !state.cameraPhotosVisible; if (!state.cameraPhotosVisible && cameraCard) cameraCard.classList.remove('is-open'); saveCameraSettings(); syncCameraControls(); draw(); }};
    const cameraColourButton = root.querySelector('[data-action="camera-colour"]');
    if (cameraColourButton) cameraColourButton.onclick = () => {{ state.cameraErrorColours = !state.cameraErrorColours; cameraColourButton.classList.toggle('is-selected', state.cameraErrorColours); draw(); }};
    root.querySelector('[data-action="zoom-in"]').onclick = () => {{ state.zoom = Math.min(12, state.zoom * 1.25); draw(); }};
    root.querySelector('[data-action="zoom-out"]').onclick = () => {{ state.zoom = Math.max(.25, state.zoom / 1.25); draw(); }};
    root.querySelector('[data-action="fullscreen"]').onclick = () => {{ if (root.requestFullscreen) root.requestFullscreen(); }};
    root.querySelectorAll('[data-view]').forEach(button => button.onclick = () => {{
      state.viewMode = button.dataset.view;
      if (state.viewMode === 'terrain') {{ state.yaw = 0; state.terrainTilt = 0; }}
      else if (state.yaw === 0 && state.pitch === 0) {{ state.yaw = .65; state.pitch = -.45; }}
      root.querySelectorAll('[data-view]').forEach(item => item.classList.toggle('is-selected', item === button));
      fitAll();
    }});
    root.querySelectorAll('[data-mode]').forEach(button => button.onclick = () => {{ state.mode = button.dataset.mode; root.querySelectorAll('[data-mode]').forEach(item => item.classList.toggle('is-selected', item === button)); draw(); }});
    root.querySelector('[data-action="size"]').oninput = event => {{ state.pointSize = Number(event.target.value); draw(); }};
    [['camera-size', 'cameraSize'], ['camera-path-width', 'cameraPathWidth'], ['camera-photo-size', 'cameraPhotoSize']].forEach(([action, key]) => {{ const control = root.querySelector(`[data-action="${{action}}"]`); if (control) control.oninput = event => {{ state[key] = Number(event.target.value); saveCameraSettings(); draw(); }}; }});
    syncCameraControls();
    canvas.oncontextmenu = event => event.preventDefault();
    canvas.onpointerdown = event => {{
      if (event.button !== 0 && event.button !== 2) return;
      if (event.button === 0 && (selectCamera(event) || selectMarker(event))) return;
      state.dragging = true; state.dragMode = event.button === 2 ? 'pan' : 'rotate';
      if (state.dragMode === 'rotate') setRotationPivot(event.clientX, event.clientY);
      state.lastX = event.clientX; state.lastY = event.clientY; canvas.classList.add('is-dragging'); canvas.setPointerCapture(event.pointerId);
    }};
    canvas.onpointermove = event => {{
      if (!state.dragging) return;
      const dx = event.clientX - state.lastX, dy = event.clientY - state.lastY;
      if (state.dragMode === 'pan') {{ state.panX += dx; state.panY += dy; }}
      else if (state.viewMode === 'terrain') {{ state.yaw += dx * .008; state.terrainTilt = Math.max(0, Math.min(1.25, state.terrainTilt - dy * .008)); }}
      else {{ state.yaw += dx * .008; state.pitch -= dy * .008; state.pitch = Math.max(-1.5, Math.min(1.5, state.pitch)); }}
      state.lastX = event.clientX; state.lastY = event.clientY; draw();
    }};
    canvas.onpointerup = () => {{ state.dragging = false; state.dragMode = ''; canvas.classList.remove('is-dragging'); }};
    canvas.onpointercancel = () => {{ state.dragging = false; state.dragMode = ''; canvas.classList.remove('is-dragging'); }};
    canvas.onwheel = event => {{ event.preventDefault(); state.zoom = Math.max(.25, Math.min(12, state.zoom * (event.deltaY < 0 ? 1.15 : .87))); draw(); }};
    window.addEventListener('resize', resize); resize(); load();
  }};
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, {{ once: true }}); else setTimeout(init, 0);
}})();
</script>
"""
