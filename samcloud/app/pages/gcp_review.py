"""Agisoft-style browser controls for reviewing automatic control points."""

from nicegui import ui


def render_review_workspace() -> None:
    """Render the marker table, filtered photo strip, and pixel-level viewer."""
    with ui.card().classes("w-full"):
        ui.label("Passpunkt- und Kontrollpunktprüfung").classes("text-lg font-medium")
        ui.html(_workspace_html(), sanitize=False).classes("w-full")
    ui.add_body_html(_workspace_script())


def _workspace_html() -> str:
    return """
<style>
.sam-gcp-review { color:#dcecf6; }.sam-gcp-banner { margin:8px 0 14px; padding:10px 12px; border:1px solid #47718a; border-radius:6px; background:#102131; }.sam-gcp-banner.failed { border-color:#ff5a5a; }.sam-gcp-actions { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px; }.sam-gcp-actions button,.sam-gcp-table button { border:1px solid #47718a; background:#102131; color:#dcecf6; border-radius:5px; padding:5px 8px; cursor:pointer; }.sam-gcp-actions button.primary { border-color:#26b7ef; background:#16516c; }.sam-gcp-grid { display:grid; grid-template-columns:minmax(520px,1fr) minmax(360px,1fr); gap:14px; }.sam-gcp-table-wrap { overflow:auto; max-height:calc(45vh - 40px); min-height:260px; border:1px solid #28475b; border-radius:6px; }.sam-gcp-table { width:100%; border-collapse:collapse; font-size:12px; }.sam-gcp-table th,.sam-gcp-table td { border-bottom:1px solid #28475b; padding:6px; text-align:right; white-space:nowrap; }.sam-gcp-table th:first-child,.sam-gcp-table td:first-child,.sam-gcp-table td:nth-child(3) { text-align:left; }.sam-gcp-table tr { cursor:pointer; }.sam-gcp-table tr.selected { background:#16516c; }.sam-gcp-table th { position:sticky; top:0; background:#102131; }.sam-gcp-photos { min-height:260px; max-height:calc(45vh - 40px); overflow:auto; border:1px solid #28475b; border-radius:6px; padding:8px; }.sam-gcp-photos-head { display:flex; justify-content:space-between; gap:8px; }.sam-gcp-strip { display:flex; gap:9px; overflow:auto; padding-top:8px; }.sam-gcp-thumb { width:clamp(150px,11vw,240px); flex:0 0 clamp(150px,11vw,240px); border:1px solid #28475b; border-radius:5px; padding:5px; background:#050b12; cursor:pointer; text-align:left; }.sam-gcp-thumb img { width:100%; image-rendering:pixelated; display:block; }.sam-gcp-thumb.muted { opacity:.45; }.sam-gcp-thumb small { display:block; margin-top:4px; color:#b9ccda; }.sam-gcp-viewer { margin-top:14px; border:1px solid #28475b; border-radius:6px; padding:9px; }.sam-gcp-photo-canvas { height:max(430px, calc(100vh - 140px)); overflow:hidden; background:#050b12; position:relative; cursor:crosshair; user-select:none; }.sam-gcp-photo-canvas img { image-rendering:pixelated; }.sam-gcp-info { color:#b9ccda; font-size:12px; margin-bottom:7px; }.sam-gcp-empty { color:#b9ccda; padding:16px 0; } @media (max-width:900px) { .sam-gcp-grid { grid-template-columns:1fr; } }
</style>
<section id="sam-gcp-review" class="sam-gcp-review">
  <div class="sam-gcp-banner" data-gate>Kontrollpunkte werden geladen.</div>
  <div class="sam-gcp-actions"><button class="primary" data-optimize>Optimieren</button><button data-continue>Weiter verarbeiten</button><button data-reload>Aktualisieren</button></div>
  <div class="sam-gcp-grid">
    <div class="sam-gcp-table-wrap"><table class="sam-gcp-table"><thead><tr><th>Aktiv</th><th>Name</th><th>Rolle</th><th>X</th><th>Y</th><th>Z</th><th>Fehler X</th><th>Fehler Y</th><th>Fehler Z</th><th>Fehler</th><th>Projektionen</th><th>Fehler px</th></tr></thead><tbody data-table></tbody></table></div>
    <div class="sam-gcp-photos"><div class="sam-gcp-photos-head"><strong data-title>Fotos nach Marker filtern</strong><button data-sort>Nach Restfehler sortieren</button></div><div class="sam-gcp-strip" data-photos></div></div>
  </div>
  <div class="sam-gcp-viewer"><div class="sam-gcp-info" data-info>Ein Foto auswählen. Mausrad: Zoom, ziehen: Verschieben, Klick: Marker setzen. Pfeiltasten wechseln Fotos, Entf entfernt die Projektion, Leertaste aktiviert/deaktiviert sie.</div><div class="sam-gcp-photo-canvas" data-viewer></div></div>
</section>
"""


def _workspace_script() -> str:
    return """
<script>
(() => {
  const init = () => {
    const root = document.getElementById('sam-gcp-review');
    if (!root || root.dataset.ready === '1') return;
    root.dataset.ready = '1';

    const table = root.querySelector('[data-table]');
    const photos = root.querySelector('[data-photos]');
    const viewer = root.querySelector('[data-viewer]');
    const gate = root.querySelector('[data-gate]');
    const info = root.querySelector('[data-info]');

    const state = {
      markers: [],
      requestedHandled: false,
      selected: '',
      photos: [],
      current: 0,
      descending: true,
      zoom: 1,
      panX: 0,
      panY: 0,
      dragging: false,
      moved: false,
      startX: 0,
      startY: 0,
      lastX: 0,
      lastY: 0,
      cropCenterU: null,
      cropCenterV: null,
      viewerOriginU: 0,
      viewerOriginV: 0,
      viewerCropSize: 512,
      currentU: null,
      currentV: null,
      viewerUrl: null,
      thumbUrls: [],
      lastPhotoKey: '',
      wrapper: null
    };

    const fitView = () => {
      const width = viewer.clientWidth || 512, height = viewer.clientHeight || 512;
      state.zoom = Math.max(1, Math.min(16, Math.min(width, height) / 512));
      state.panX = (width - 512 * state.zoom) / 2;
      state.panY = (height - 512 * state.zoom) / 2;
    };

    const post = async (url, value = {}) => {
      const response = await fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(value)
      });
      if (!response.ok) {
        throw new Error((await response.json()).error || 'Aktion fehlgeschlagen');
      }
      return response.json();
    };

    const number = value => value === null || value === undefined ? '–' : Number(value).toFixed(3);

    const cropUrl = (photo, size = 96, centerU = null, centerV = null) => {
      const u = centerU !== null && centerU !== undefined ? centerU : photo.u;
      const v = centerV !== null && centerV !== undefined ? centerV : photo.v;
      return '/api/control-points/crop?' + new URLSearchParams({
        image: photo.image,
        u: String(u),
        v: String(v),
        size: String(size),
        source: 'original'
      });
    };

    const marker = () => state.markers.find(item => item.name === state.selected);

    const getPhotos = () => {
      const item = marker();
      if (!item) return [];
      const list = item.photos || item.observations || [];
      const detected = list.filter(p => p.kind !== 'predicted');
      const predicted = list.filter(p => p.kind === 'predicted');
      const sortFn = (a, b) => {
        const ra = a.reprojection_px !== null && a.reprojection_px !== undefined ? a.reprojection_px : -1;
        const rb = b.reprojection_px !== null && b.reprojection_px !== undefined ? b.reprojection_px : -1;
        return state.descending ? rb - ra : ra - rb;
      };
      detected.sort(sortFn);
      return [...detected, ...predicted];
    };

    const renderSvgOverlay = (photo, originU, originV, S) => {
      const r = Math.max(4, Math.min(14, S * 0.025));
      const elements = [];
      if (photo.kind === 'predicted') {
        const px = photo.u - originU;
        const py = photo.v - originV;
        const cr = r * 1.3;
        elements.push(`<circle cx="${px}" cy="${py}" r="${cr}" fill="none" stroke="#8a9ba8" stroke-dasharray="3,3" stroke-width="1.5" vector-effect="non-scaling-stroke" style="vector-effect:non-scaling-stroke;" />`);
        elements.push(`<line x1="${px - r}" y1="${py}" x2="${px + r}" y2="${py}" stroke="#8a9ba8" stroke-width="1.5" vector-effect="non-scaling-stroke" style="vector-effect:non-scaling-stroke;" />`);
        elements.push(`<line x1="${px}" y1="${py - r}" x2="${px}" y2="${py + r}" stroke="#8a9ba8" stroke-width="1.5" vector-effect="non-scaling-stroke" style="vector-effect:non-scaling-stroke;" />`);
      } else {
        const color = photo.enabled === false ? '#6b7c8a' : (photo.kind === 'manual' ? '#4f8cff' : '#3ddc84');
        const x = photo.u - originU;
        const y = photo.v - originV;
        const h = r * 2.2;
        const w = r * 1.6;
        const f = r * 1.1;
        elements.push(`<line x1="${x - r}" y1="${y}" x2="${x + r}" y2="${y}" stroke="${color}" stroke-width="1.5" vector-effect="non-scaling-stroke" style="vector-effect:non-scaling-stroke;" />`);
        elements.push(`<line x1="${x}" y1="${y - r}" x2="${x}" y2="${y + r}" stroke="${color}" stroke-width="1.5" vector-effect="non-scaling-stroke" style="vector-effect:non-scaling-stroke;" />`);
        elements.push(`<line x1="${x}" y1="${y}" x2="${x}" y2="${y - h}" stroke="${color}" stroke-width="1.5" vector-effect="non-scaling-stroke" style="vector-effect:non-scaling-stroke;" />`);
        elements.push(`<polygon points="${x},${y - h} ${x + w},${y - h + f * 0.5} ${x},${y - h + f}" fill="${color}" stroke="${color}" stroke-width="1" vector-effect="non-scaling-stroke" style="vector-effect:non-scaling-stroke;" />`);

        if (photo.reprojected_u !== null && photo.reprojected_u !== undefined &&
            photo.reprojected_v !== null && photo.reprojected_v !== undefined &&
            !isNaN(photo.reprojected_u) && !isNaN(photo.reprojected_v)) {
          const rx = photo.reprojected_u - originU;
          const ry = photo.reprojected_v - originV;
          const xr = r * 0.6;
          elements.push(`<line x1="${rx - xr}" y1="${ry - xr}" x2="${rx + xr}" y2="${ry + xr}" stroke="#ffb020" stroke-width="1.5" vector-effect="non-scaling-stroke" style="vector-effect:non-scaling-stroke;" />`);
          elements.push(`<line x1="${rx - xr}" y1="${ry + xr}" x2="${rx + xr}" y2="${ry - xr}" stroke="#ffb020" stroke-width="1.5" vector-effect="non-scaling-stroke" style="vector-effect:non-scaling-stroke;" />`);
        }
      }
      return `<svg viewBox="0 0 ${S} ${S}" style="position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none;">${elements.join('')}</svg>`;
    };

    const renderGate = value => {
      const accepted = value.gate && value.gate.passed;
      gate.classList.toggle('failed', !accepted);
      const reasons = (value.gate && value.gate.reasons || []).join('; ');
      gate.textContent = accepted ? 'Qualitätsprüfung bestanden.' : `Qualitätsprüfung fehlgeschlagen${reasons ? ': ' + reasons : ''}`;
    };

    const renderTable = () => {
      table.replaceChildren();
      for (const item of state.markers) {
        const row = document.createElement('tr');
        if (item.name === state.selected) row.classList.add('selected');
        row.onclick = () => {
          state.selected = item.name;
          state.current = 0;
          fitView();
          state.cropCenterU = null;
          state.cropCenterV = null;
          state.lastPhotoKey = '';
          render();
          root.querySelector('[data-photos]').scrollIntoView({behavior: 'smooth', block: 'nearest'});
        };
        const e = item.errors || {};
        row.innerHTML = `<td><input type="checkbox" ${item.enabled ? 'checked' : ''}></td><td>${item.name}</td><td><button>${item.role === 'pass' ? 'Passpunkt' : 'Kontrollpunkt'}</button></td><td>${number(item.coordinates && item.coordinates[0])}</td><td>${number(item.coordinates && item.coordinates[1])}</td><td>${number(item.coordinates && item.coordinates[2])}</td><td>${number(e.x_m)}</td><td>${number(e.y_m)}</td><td>${number(e.z_m)}</td><td>${number(e.total_m)}</td><td>${item.projections}</td><td>${number(e.reprojection_px)}</td>`;
        const enabled = row.querySelector('input');
        enabled.onclick = event => event.stopPropagation();
        enabled.onchange = async () => {
          try {
            await post('/api/control-points/marker/toggle', {name: item.name, enabled: enabled.checked});
            await load();
          } catch (error) {
            info.textContent = error.message;
          }
        };
        const role = row.querySelector('button');
        role.onclick = async event => {
          event.stopPropagation();
          try {
            await post('/api/control-points/marker/role', {name: item.name, role: item.role === 'pass' ? 'control' : 'pass'});
            await load();
          } catch (error) {
            info.textContent = error.message;
          }
        };
        table.append(row);
      }
    };

    const renderPhotos = () => {
      state.thumbUrls.forEach(url => URL.revokeObjectURL(url));
      state.thumbUrls = [];
      photos.replaceChildren();

      const prevPhoto = state.photos[state.current];
      state.photos = getPhotos();
      if (prevPhoto) {
        const idx = state.photos.findIndex(p => p.image === prevPhoto.image);
        state.current = idx !== -1 ? idx : 0;
      } else {
        state.current = 0;
      }

      const active = marker();
      root.querySelector('[data-title]').textContent = active ? `Fotos: ${active.name}` : 'Fotos nach Marker filtern';
      if (!state.photos.length) {
        photos.innerHTML = '<div class="sam-gcp-empty">Für diesen Marker sind keine Projektionen gespeichert.</div>';
        return;
      }

      state.photos.forEach((photo, index) => {
        const thumb = document.createElement('button');
        thumb.className = 'sam-gcp-thumb' + (photo.kind === 'predicted' || photo.enabled === false ? ' muted' : '');
        if (index === state.current) {
          thumb.style.borderColor = '#26b7ef';
        }

        const mediaWrap = document.createElement('div');
        mediaWrap.style.cssText = 'position:relative;width:100%;aspect-ratio:1/1;background:#050b12;overflow:hidden;border-radius:3px;';

        const label1 = document.createElement('small');
        label1.textContent = photo.image;

        const label2 = document.createElement('small');
        if (photo.kind === 'predicted') {
          label2.textContent = 'nicht erkannt';
        } else {
          const kindLabel = photo.kind === 'manual' ? 'manuell' : 'automatisch';
          label2.textContent = `${number(photo.reprojection_px)} px · ${kindLabel}`;
        }

        thumb.append(mediaWrap, label1, label2);

        thumb.onclick = () => {
          state.current = index;
          const allThumbs = photos.querySelectorAll('.sam-gcp-thumb');
          allThumbs.forEach((t, i) => t.style.borderColor = i === index ? '#26b7ef' : '#28475b');
          fitView();
          state.cropCenterU = photo.u;
          state.cropCenterV = photo.v;
          state.lastPhotoKey = photo.image;
          renderViewer();
        };

        photos.append(thumb);

        fetch(cropUrl(photo, 96))
          .then(async res => {
            if (!res.ok) return;
            const originU = parseFloat(res.headers.get('X-Crop-Origin-U') || '0');
            const originV = parseFloat(res.headers.get('X-Crop-Origin-V') || '0');
            const cropSize = parseFloat(res.headers.get('X-Crop-Size') || res.headers.get('X-SAMcloud-Crop-Size') || '96');
            const blob = await res.blob();
            const objUrl = URL.createObjectURL(blob);
            state.thumbUrls.push(objUrl);

            const img = document.createElement('img');
            img.src = objUrl;
            img.alt = photo.image;
            img.style.cssText = 'width:100%;height:100%;display:block;image-rendering:pixelated;';

            const overlaySvg = renderSvgOverlay(photo, originU, originV, cropSize);
            mediaWrap.replaceChildren(img);
            mediaWrap.insertAdjacentHTML('beforeend', overlaySvg);
          })
          .catch(() => {});
      });
    };

    const updateInfoText = () => {
      const photo = state.photos[state.current];
      if (!photo) return;
      const liveU = state.currentU !== null && !isNaN(state.currentU) ? state.currentU.toFixed(2) : Number(photo.u).toFixed(2);
      const liveV = state.currentV !== null && !isNaN(state.currentV) ? state.currentV.toFixed(2) : Number(photo.v).toFixed(2);
      const reproj = photo.reprojection_px !== null && photo.reprojection_px !== undefined ? `, Restfehler ${number(photo.reprojection_px)} px` : '';
      const score = photo.score !== null && photo.score !== undefined ? `, Score ${number(photo.score)}` : '';
      info.textContent = `${photo.image}: u ${liveU}, v ${liveV}${reproj}${score}. Zoom ${state.zoom.toFixed(1)}×`;
    };

    const renderViewerDOM = () => {
      const photo = state.photos[state.current];
      if (!photo) return;
      viewer.replaceChildren();

      const wrapper = document.createElement('div');
      wrapper.style.cssText = `position:absolute;left:0;top:0;width:512px;height:512px;transform-origin:0 0;will-change:transform;transform:translate(${state.panX}px, ${state.panY}px) scale(${state.zoom});`;

      const img = document.createElement('img');
      img.src = state.viewerUrl;
      img.alt = photo.image;
      img.draggable = false;
      img.style.cssText = 'position:absolute;left:0;top:0;width:100%;height:100%;display:block;image-rendering:pixelated;user-select:none;pointer-events:none;';

      const overlaySvg = renderSvgOverlay(photo, state.viewerOriginU, state.viewerOriginV, state.viewerCropSize);

      wrapper.append(img);
      wrapper.insertAdjacentHTML('beforeend', overlaySvg);
      viewer.append(wrapper);
      state.wrapper = wrapper;

      updateInfoText();
    };

    const loadViewerCrop = async (keepView = false) => {
      const photo = state.photos[state.current];
      const selected = marker();
      if (!selected || !photo) {
        viewer.replaceChildren();
        return;
      }

      const url = cropUrl(photo, 512, state.cropCenterU, state.cropCenterV);
      try {
        const res = await fetch(url);
        if (!res.ok) return;

        const originU = parseFloat(res.headers.get('X-Crop-Origin-U') || '0');
        const originV = parseFloat(res.headers.get('X-Crop-Origin-V') || '0');
        const cropSize = parseFloat(res.headers.get('X-Crop-Size') || res.headers.get('X-SAMcloud-Crop-Size') || '512');

        state.viewerOriginU = originU;
        state.viewerOriginV = originV;
        state.viewerCropSize = cropSize;

        const blob = await res.blob();
        if (state.viewerUrl) {
          URL.revokeObjectURL(state.viewerUrl);
        }
        state.viewerUrl = URL.createObjectURL(blob);

        const canvasW = viewer.clientWidth || 512;
        const canvasH = viewer.clientHeight || 430;

        if (!keepView) {
          const markerX = ((photo.u - originU) / cropSize) * 512;
          const markerY = ((photo.v - originV) / cropSize) * 512;
          state.panX = canvasW / 2 - markerX * state.zoom;
          state.panY = canvasH / 2 - markerY * state.zoom;
        } else {
          const centerX = ((state.cropCenterU - originU) / cropSize) * 512;
          const centerY = ((state.cropCenterV - originV) / cropSize) * 512;
          state.panX = canvasW / 2 - centerX * state.zoom;
          state.panY = canvasH / 2 - centerY * state.zoom;
        }

        renderViewerDOM();
      } catch (err) {}
    };

    const renderViewer = () => {
      const selected = marker();
      const photo = state.photos[state.current];
      if (!selected || !photo) {
        viewer.replaceChildren();
        info.textContent = 'Ein Foto auswählen. Mausrad: Zoom, ziehen: Verschieben, Klick: Marker setzen. Pfeiltasten wechseln Fotos, Entf entfernt die Projektion, Leertaste aktiviert/deaktiviert sie.';
        return;
      }

      if (state.lastPhotoKey !== photo.image) {
        state.lastPhotoKey = photo.image;
        state.cropCenterU = photo.u;
        state.cropCenterV = photo.v;
        fitView();
        state.currentU = null;
        state.currentV = null;
      }

      loadViewerCrop(false);
    };

    viewer.onpointerdown = event => {
      if (event.button !== 0) return;
      state.dragging = true;
      state.moved = false;
      state.startX = event.clientX;
      state.startY = event.clientY;
      state.lastX = event.clientX;
      state.lastY = event.clientY;
      try { viewer.setPointerCapture(event.pointerId); } catch (e) {}
    };

    viewer.onpointermove = event => {
      const canvasRect = viewer.getBoundingClientRect();
      const mouseCanvasX = event.clientX - canvasRect.left;
      const mouseCanvasY = event.clientY - canvasRect.top;
      const offsetX = (mouseCanvasX - state.panX) / state.zoom;
      const offsetY = (mouseCanvasY - state.panY) / state.zoom;
      state.currentU = state.viewerOriginU + (offsetX / 512) * state.viewerCropSize;
      state.currentV = state.viewerOriginV + (offsetY / 512) * state.viewerCropSize;

      if (state.dragging) {
        const dx = event.clientX - state.lastX;
        const dy = event.clientY - state.lastY;
        if (Math.hypot(event.clientX - state.startX, event.clientY - state.startY) > 3) {
          state.moved = true;
        }
        state.panX += dx;
        state.panY += dy;
        state.lastX = event.clientX;
        state.lastY = event.clientY;
        if (state.wrapper) {
          state.wrapper.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.zoom})`;
        }
      }
      updateInfoText();
    };

    viewer.onpointerup = async event => {
      if (!state.dragging) return;
      state.dragging = false;
      try { viewer.releasePointerCapture(event.pointerId); } catch (e) {}

      const photo = state.photos[state.current];
      const selected = marker();
      if (!selected || !photo) return;

      if (!state.moved) {
        const canvasRect = viewer.getBoundingClientRect();
        const mouseCanvasX = event.clientX - canvasRect.left;
        const mouseCanvasY = event.clientY - canvasRect.top;
        const offsetX = (mouseCanvasX - state.panX) / state.zoom;
        const offsetY = (mouseCanvasY - state.panY) / state.zoom;
        const clickU = state.viewerOriginU + (offsetX / 512) * state.viewerCropSize;
        const clickV = state.viewerOriginV + (offsetY / 512) * state.viewerCropSize;

        try {
          await post('/api/control-points/observation', {
            name: selected.name,
            image: photo.image,
            u: clickU,
            v: clickV
          });
          state.cropCenterU = clickU;
          state.cropCenterV = clickV;
          await load();
        } catch (error) {
          info.textContent = error.message;
        }
        return;
      }

      const canvasW = viewer.clientWidth || 512;
      const canvasH = viewer.clientHeight || 430;
      const viewCenterOffsetX = (canvasW / 2 - state.panX) / state.zoom;
      const viewCenterOffsetY = (canvasH / 2 - state.panY) / state.zoom;
      const viewCenterU = state.viewerOriginU + (viewCenterOffsetX / 512) * state.viewerCropSize;
      const viewCenterV = state.viewerOriginV + (viewCenterOffsetY / 512) * state.viewerCropSize;

      const dist = Math.hypot(viewCenterU - state.cropCenterU, viewCenterV - state.cropCenterV);
      if (dist > 128) {
        state.cropCenterU = viewCenterU;
        state.cropCenterV = viewCenterV;
        await loadViewerCrop(true);
      }
    };

    viewer.onwheel = event => {
      event.preventDefault();
      const canvasRect = viewer.getBoundingClientRect();
      const mouseCanvasX = event.clientX - canvasRect.left;
      const mouseCanvasY = event.clientY - canvasRect.top;

      const wrapperX = (mouseCanvasX - state.panX) / state.zoom;
      const wrapperY = (mouseCanvasY - state.panY) / state.zoom;

      const factor = event.deltaY < 0 ? 1.25 : 0.8;
      const newZoom = Math.max(1, Math.min(16, state.zoom * factor));
      if (newZoom === state.zoom) return;

      state.panX = mouseCanvasX - wrapperX * newZoom;
      state.panY = mouseCanvasY - wrapperY * newZoom;
      state.zoom = newZoom;

      if (state.wrapper) {
        state.wrapper.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.zoom})`;
      }

      const offsetX = (mouseCanvasX - state.panX) / state.zoom;
      const offsetY = (mouseCanvasY - state.panY) / state.zoom;
      state.currentU = state.viewerOriginU + (offsetX / 512) * state.viewerCropSize;
      state.currentV = state.viewerOriginV + (offsetY / 512) * state.viewerCropSize;
      updateInfoText();
    };

    viewer.onpointerleave = () => {
      state.currentU = null;
      state.currentV = null;
      updateInfoText();
    };

    const render = () => {
      renderTable();
      renderPhotos();
      renderViewer();
    };

    const load = async () => {
      try {
        const response = await fetch('/api/control-points/markers', {cache: 'no-store'});
        if (!response.ok) throw new Error((await response.json()).error || 'Marker konnten nicht geladen werden');
        const value = await response.json();
        state.markers = value.markers || [];
        if (!state.requestedHandled) {
          state.requestedHandled = true;
          const requested = new URLSearchParams(window.location.search).get('marker');
          if (requested && state.markers.some(item => item.name === requested)) {
            state.selected = requested;
            setTimeout(() => root.scrollIntoView({behavior: 'smooth', block: 'start'}), 50);
          }
        }
        if (!state.markers.some(item => item.name === state.selected)) { const first = state.markers.find(item => (item.photos || []).some(photo => photo.kind !== 'predicted')) || state.markers[0]; state.selected = first ? first.name : ''; }
        renderGate(value);
        render();
      } catch (error) {
        gate.classList.add('failed');
        gate.textContent = error.message;
      }
    };

    root.querySelector('[data-sort]').onclick = () => {
      state.descending = !state.descending;
      renderPhotos();
      renderViewer();
    };

    root.querySelector('[data-reload]').onclick = load;

    root.querySelector('[data-optimize]').onclick = async () => {
      try {
        await post('/api/control-points/optimize');
        info.textContent = 'Optimierung läuft. Die Werte werden nach Abschluss aktualisiert.';
      } catch (error) {
        info.textContent = error.message;
      }
    };

    root.querySelector('[data-continue]').onclick = async () => {
      try {
        await post('/api/control-points/continue');
        info.textContent = 'Verarbeitung wurde fortgesetzt.';
      } catch (error) {
        info.textContent = error.message;
      }
    };

    window.addEventListener('samcloud-marker-selected', event => {
      if (!event.detail || !state.markers.some(item => item.name === event.detail.name)) return;
      state.selected = event.detail.name;
      state.current = 0;
      fitView();
      state.lastPhotoKey = '';
      render();
    });

    window.addEventListener('keydown', async event => {
      if (!state.photos.length || ['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
      if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
        event.preventDefault();
        state.current = (state.current + (event.key === 'ArrowRight' ? 1 : state.photos.length - 1)) % state.photos.length;
        const thumbs = photos.querySelectorAll('.sam-gcp-thumb');
        thumbs.forEach((t, i) => t.style.borderColor = i === state.current ? '#26b7ef' : '#28475b');
        if (thumbs[state.current]) {
          thumbs[state.current].scrollIntoView({ block: 'nearest', inline: 'nearest' });
        }
        fitView();
        renderViewer();
        return;
      }
      const selected = marker(), photo = state.photos[state.current];
      if (!selected || !photo) return;
      if (photo.kind === 'predicted') return;
      try {
        if (event.key === 'Delete') {
          event.preventDefault();
          await post('/api/control-points/observation/remove', {name: selected.name, image: photo.image});
          await load();
        }
        if (event.key === ' ') {
          event.preventDefault();
          await post('/api/control-points/observation/toggle', {name: selected.name, image: photo.image, enabled: !photo.enabled});
          await load();
        }
      } catch (error) {
        info.textContent = error.message;
      }
    });

    load();
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, {once: true});
  } else {
    setTimeout(init, 0);
  }
})();
</script>
"""
