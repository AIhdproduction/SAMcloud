/* DXF preview viewer (outdoor DXF/IFC page).
 * Vector mode (.sam-dxfp-svg): the inline SVG is zoomed by rewriting its viewBox, so lines and text are re-rendered
 * as vectors at every zoom level. Raster mode (.sam-dxfp): CSS transform on a stack of PNG layers (fallback).
 * AI assistant (only while the page sets data-sam-pick="1" and data-sam-origin="[ox, oy]" on the viewport): a click
 * (pointer moved less than CLICK_PX) dispatches the DOM event "sampick" {e, n, sx, sy, ctrl} with LV95 E/N
 * (E = ox + sx, N = oy - sy; sx/sy from the viewBox at pointer-down), and samDxfpOverlay draws the selection / proposal
 * rings (LV95) in a <g class="sam-dxfp-overlay"> above the drawing. Without those attributes nothing changes. */
(function () {
  var MIN_UNITS = 0.15;                     // smallest visible width in metres (single tree label)
  var CLICK_PX = 5;                         // pointer travel below this is a click (pick), above it a pan
  var SVGNS = 'http://www.w3.org/2000/svg';
  var OVERLAYS = {};                        // viewport id -> {selection: [...], preview: [...]} (LV95 items)

  function rawView(vp) { if (!vp._v) vp._v = {k: 1, x: 0, y: 0}; return vp._v; }
  function applyRaster(vp) {
    var v = rawView(vp);
    vp.firstElementChild.style.transform = 'translate(' + v.x + 'px,' + v.y + 'px) scale(' + v.k + ')';
  }

  function svgOf(vp) { return vp._svg || null; }
  function setBox(vp) {
    var b = vp._box, svg = svgOf(vp);
    if (svg) svg.setAttribute('viewBox', b.x + ' ' + b.y + ' ' + b.w + ' ' + b.h);
  }
  function size(vp) { var r = vp.getBoundingClientRect(); return {w: Math.max(r.width, 1), h: Math.max(r.height, 1), left: r.left, top: r.top}; }
  function fitSvg(vp) {
    var base = vp._base, s = size(vp); if (!base) return;
    var k = Math.max(base.w / s.w, base.h / s.h);              // metres per pixel so that everything fits
    var w = s.w * k, h = s.h * k;
    vp._box = {x: base.x + base.w / 2 - w / 2, y: base.y + base.h / 2 - h / 2, w: w, h: h};
    setBox(vp);
  }

  window.samDxfpLoad = function (id, url) {
    var vp = document.getElementById(id); if (!vp) return Promise.resolve(false);
    return fetch(url, {cache: 'no-cache'}).then(function (r) { return r.text(); }).then(function (text) {
      var host = vp.querySelector('.sam-dxfp-svgbox'); host.innerHTML = text;
      var svg = host.firstElementChild; if (!svg || svg.tagName.toLowerCase() !== 'svg') return false;
      svg.removeAttribute('width'); svg.removeAttribute('height');
      var vb = svg.viewBox.baseVal;
      vp._svg = svg; vp._base = {x: vb.x, y: vb.y, w: vb.width, h: vb.height};
      fitSvg(vp);
      if (window.ResizeObserver && !vp._ro) {
        vp._ro = new ResizeObserver(function () {
          if (!vp._box) return; var s = size(vp);
          vp._box.h = vp._box.w * s.h / s.w; setBox(vp);
        });
        vp._ro.observe(vp);
      }
      vp.setAttribute('data-ready', '1');
      drawOverlay(vp);
      return true;
    });
  };

  function originOf(vp) {
    try {
      var o = JSON.parse(vp.getAttribute('data-sam-origin') || 'null');
      return (o && o.length >= 2) ? [Number(o[0]), Number(o[1])] : null;
    } catch (err) { return null; }
  }

  function overlayPath(points, origin) {
    var d = '';
    for (var i = 0; i < points.length; i++) {
      var p = points[i];
      d += (i ? 'L' : 'M') + (Number(p[0]) - origin[0]) + ' ' + (origin[1] - Number(p[1]));
    }
    return d + 'Z';
  }

  function drawItem(g, item, origin) {
    var color = item.color || '#26b7ef', node;
    if (item.ring && item.ring.length > 1) {
      node = document.createElementNS(SVGNS, 'path');
      node.setAttribute('d', overlayPath(item.ring, origin));
      node.style.fill = item.fill ? color : 'none';
      if (item.fill) node.style.fillOpacity = String(item.fill);
    } else if (item.point && item.point.length >= 2) {
      var x = Number(item.point[0]) - origin[0], y = origin[1] - Number(item.point[1]), r = 0.35;
      node = document.createElementNS(SVGNS, 'path');
      node.setAttribute('d', 'M' + (x - r) + ' ' + y + 'L' + (x + r) + ' ' + y + 'M' + x + ' ' + (y - r) + 'L' + x + ' ' + (y + r) +
        'M' + (x + r) + ' ' + y + 'A' + r + ' ' + r + ' 0 1 0 ' + (x - r) + ' ' + y + 'A' + r + ' ' + r + ' 0 1 0 ' + (x + r) + ' ' + y);
      node.style.fill = 'none';
    } else { return; }
    // inline styles: the drawing's own stylesheet (.samdxfp-svg path) would override presentation attributes
    node.style.stroke = color;
    node.style.strokeWidth = String(item.width || 2.5);
    node.style.vectorEffect = 'non-scaling-stroke';
    node.style.strokeLinejoin = 'round';
    if (item.dashed) node.style.strokeDasharray = '6 4';
    g.appendChild(node);
  }

  function drawOverlay(vp) {
    var svg = svgOf(vp); if (!svg) return;
    var old = svg.querySelector('g.sam-dxfp-overlay'); if (old) old.parentNode.removeChild(old);
    var data = OVERLAYS[vp.id], origin = originOf(vp); if (!data || !origin) return;
    var g = document.createElementNS(SVGNS, 'g');
    g.setAttribute('class', 'sam-dxfp-overlay');
    g.setAttribute('pointer-events', 'none');
    var groups = [data.preview || [], data.selection || []];   // the selection is drawn on top of the proposal colours
    for (var k = 0; k < groups.length; k++) for (var i = 0; i < groups[k].length; i++) drawItem(g, groups[k][i], origin);
    svg.appendChild(g);
  }

  window.samDxfpOverlay = function (id, data) {
    OVERLAYS[id] = data || {};
    var vp = document.getElementById(id); if (vp) drawOverlay(vp);
  };

  window.samDxfpFit = function (id) {
    var vp = document.getElementById(id); if (!vp) return;
    if (svgOf(vp)) { fitSvg(vp); return; }
    vp._v = {k: 1, x: 0, y: 0}; applyRaster(vp);
  };

  window.samDxfpLayer = function (id, name, visible) {
    var vp = document.getElementById(id), svg = vp && svgOf(vp); if (!svg) return;
    var groups = svg.querySelectorAll('g[data-layer]');
    for (var i = 0; i < groups.length; i++) {
      if (groups[i].getAttribute('data-layer') === name) groups[i].style.display = visible ? '' : 'none';
    }
  };

  window.samDxfpLayers = function (id, names, visible) {
    for (var i = 0; i < names.length; i++) window.samDxfpLayer(id, names[i], visible);
  };

  function zoomSvg(vp, e) {
    var b = vp._box, s = size(vp), px = e.clientX - s.left, py = e.clientY - s.top;
    var f = Math.exp(-e.deltaY * 0.0015);
    var w = Math.min(Math.max(b.w / f, MIN_UNITS), vp._base.w * 4);
    f = b.w / w;
    var ux = b.x + px * b.w / s.w, uy = b.y + py * b.h / s.h;
    var h = b.h / f;
    vp._box = {x: ux - px * w / s.w, y: uy - py * h / s.h, w: w, h: h};
    setBox(vp);
  }

  document.addEventListener('wheel', function (e) {
    var vp = e.target.closest && e.target.closest('.sam-dxfp'); if (!vp) return;
    e.preventDefault();
    if (svgOf(vp)) { zoomSvg(vp, e); return; }
    var v = rawView(vp), r = vp.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
    var k = Math.max(0.5, Math.min(200, v.k * Math.exp(-e.deltaY * 0.0015)));
    v.x = px - (px - v.x) * k / v.k; v.y = py - (py - v.y) * k / v.k; v.k = k; applyRaster(vp);
  }, {passive: false});

  var drag = null;
  document.addEventListener('pointerdown', function (e) {
    var vp = e.target.closest && e.target.closest('.sam-dxfp'); if (!vp || e.button !== 0) return;
    drag = {vp: vp, sx: e.clientX, sy: e.clientY, ctrl: !!(e.ctrlKey || e.metaKey)};
    if (svgOf(vp)) { drag.box = {x: vp._box.x, y: vp._box.y, w: vp._box.w, h: vp._box.h}; }
    else { var v = rawView(vp); drag.ox = v.x; drag.oy = v.y; }
    vp.classList.add('is-dragging');
  });
  document.addEventListener('pointermove', function (e) {
    if (!drag) return; var vp = drag.vp;
    if (drag.box) {
      var s = size(vp), b = drag.box;
      vp._box = {x: b.x - (e.clientX - drag.sx) * b.w / s.w, y: b.y - (e.clientY - drag.sy) * b.h / s.h, w: b.w, h: b.h};
      setBox(vp);
    } else {
      var v = rawView(vp); v.x = drag.ox + e.clientX - drag.sx; v.y = drag.oy + e.clientY - drag.sy; applyRaster(vp);
    }
  });
  function pickAt(d, e) {
    // read the coordinates first (from the view at pointer-down), focus afterwards without scrolling the page
    var vp = d.vp, b = d.box, origin = originOf(vp), s = size(vp);
    if (!b || !origin) return;
    var sx = b.x + (d.sx - s.left) * b.w / s.w, sy = b.y + (d.sy - s.top) * b.h / s.h;
    vp._box = {x: b.x, y: b.y, w: b.w, h: b.h}; setBox(vp);      // undo the few pixels of panning of the click
    var detail = {e: origin[0] + sx, n: origin[1] - sy, sx: sx, sy: sy, ctrl: d.ctrl || !!(e.ctrlKey || e.metaKey)};
    vp.dispatchEvent(new CustomEvent('sampick', {detail: detail}));
    if (vp.focus) vp.focus({preventScroll: true});
  }

  document.addEventListener('pointerup', function (e) {
    if (!drag) return;
    var d = drag; drag = null;
    d.vp.classList.remove('is-dragging');
    if (d.vp.getAttribute('data-sam-pick') === '1' && svgOf(d.vp) &&
        Math.abs(e.clientX - d.sx) < CLICK_PX && Math.abs(e.clientY - d.sy) < CLICK_PX) pickAt(d, e);
  });
  document.addEventListener('dblclick', function (e) {
    var vp = e.target.closest && e.target.closest('.sam-dxfp'); if (vp) window.samDxfpFit(vp.id);
  });
})();
