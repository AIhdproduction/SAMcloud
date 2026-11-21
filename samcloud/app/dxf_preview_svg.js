/* DXF preview viewer (outdoor DXF/IFC page).
 * Vector mode (.sam-dxfp-svg): the inline SVG is zoomed by rewriting its viewBox, so lines and text are re-rendered
 * as vectors at every zoom level. Raster mode (.sam-dxfp): CSS transform on a stack of PNG layers (fallback). */
(function () {
  var MIN_UNITS = 0.15;                     // smallest visible width in metres (single tree label)

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
      return true;
    });
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
    drag = {vp: vp, sx: e.clientX, sy: e.clientY};
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
  document.addEventListener('pointerup', function () { if (drag) { drag.vp.classList.remove('is-dragging'); drag = null; } });
  document.addEventListener('dblclick', function (e) {
    var vp = e.target.closest && e.target.closest('.sam-dxfp'); if (vp) window.samDxfpFit(vp.id);
  });
})();
