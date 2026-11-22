/* SAMcloud project share: shared register bar */
(function () {
  "use strict";

  const currentScript = document.currentScript || document.querySelector('script[src$="register.js"]');
  const root = (currentScript && currentScript.getAttribute("data-root")) || "";
  const activeView = (currentScript && currentScript.getAttribute("data-view")) || "";

  const i18n = {
    de: {
      overview: "Übersicht",
      dxf: "Plan (DXF)",
      pc: "Punktwolke",
      ifc: "IFC-Modell",
      dxf_short: "DXF",
      pc_short: "Punkte",
      ifc_short: "IFC",
      dxf_desc: "Grundriss/Lageplan von oben",
      pc_desc: "Klassifizierte Punktwolke",
      ifc_desc: "BIM-Modell mit Kategorien",
      points: "Punkte",
      created_with: "Erstellt mit SAMcloud",
      loading: "Lädt …",
      level: "Ebene"
    },
    en: {
      overview: "Overview",
      dxf: "Plan (DXF)",
      pc: "Point cloud",
      ifc: "IFC model",
      dxf_short: "DXF",
      pc_short: "Points",
      ifc_short: "IFC",
      dxf_desc: "Plan view from above",
      pc_desc: "Classified point cloud",
      ifc_desc: "BIM model with categories",
      points: "points",
      created_with: "Created with SAMcloud",
      loading: "Loading …",
      level: "Level"
    }
  };

  function getStoredLevel() {
    try {
      return sessionStorage.getItem("samcloud_share_level");
    } catch (_e) {
      return null;
    }
  }

  function setStoredLevel(lvl) {
    try {
      if (lvl) {
        sessionStorage.setItem("samcloud_share_level", lvl);
      }
    } catch (_e) {}
  }

  let currentLevel = null;
  const urlParams = new URLSearchParams(window.location.search);
  const paramLevel = urlParams.get("level");
  if (paramLevel) {
    currentLevel = paramLevel;
    setStoredLevel(paramLevel);
  } else {
    currentLevel = getStoredLevel();
  }

  let navElement = null;
  let lastRegH = 0;
  let inResizeDispatch = false;

  function tFactory(lang) {
    const table = i18n[lang] || i18n.de;
    return function (key) {
      return table[key] || (i18n.de && i18n.de[key]) || key;
    };
  }

  function updateRegHeight() {
    if (!navElement) return;
    const h = navElement.getBoundingClientRect().height;
    if (h > 0) {
      const rounded = Math.round(h);
      const changed = rounded !== lastRegH;
      lastRegH = rounded;
      document.documentElement.style.setProperty("--reg-h", rounded + "px");
      // Viewers measure their stage before the bar height settles: tell them once per real change.
      if (changed && !inResizeDispatch) {
        inResizeDispatch = true;
        try { window.dispatchEvent(new Event("resize")); } finally { inResizeDispatch = false; }
      }
    }
  }

  function updateSegmentLinks(lvl) {
    if (!navElement) return;
    currentLevel = lvl || currentLevel;
    if (currentLevel) setStoredLevel(currentLevel);
    const lvlQuery = currentLevel ? "?level=" + encodeURIComponent(currentLevel) : "";
    const dxfLink = navElement.querySelector('.reg-segment[data-key="dxf"]');
    if (dxfLink) dxfLink.href = root + "dxf.html" + lvlQuery;
    const ifcLink = navElement.querySelector('.reg-segment[data-key="ifc"]');
    if (ifcLink) ifcLink.href = root + "ifc.html" + lvlQuery;
  }

  function buildBar(project, lang, t) {
    if (navElement) return;

    if (!document.body) return;
    document.body.classList.add("has-register");

    const nav = document.createElement("nav");
    nav.className = "reg-bar";
    navElement = nav;

    // Home link
    const homeLink = document.createElement("a");
    homeLink.className = "reg-home";
    homeLink.href = root + "index.html";
    homeLink.setAttribute("aria-label", t("overview"));
    homeLink.title = t("overview");
    homeLink.innerHTML =
      '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 9.5L12 3l9 6.5V20a1 1 0 0 1-1 1h-5v-6h-6v6H4a1 1 0 0 1-1-1V9.5z"/></svg>';
    nav.appendChild(homeLink);

    // Segments container
    const segments = document.createElement("div");
    segments.className = "reg-segments";

    const views = (project && Array.isArray(project.views) && project.views.length)
      ? project.views
      : ["dxf", "pc", "ifc"];

    const lvlQuery = currentLevel ? "?level=" + encodeURIComponent(currentLevel) : "";

    for (const viewKey of views) {
      let href = "#";
      let fullLabel = "";
      let shortLabel = "";

      if (viewKey === "dxf") {
        href = root + "dxf.html" + lvlQuery;
        fullLabel = t("dxf");
        shortLabel = t("dxf_short");
      } else if (viewKey === "pc") {
        href = root + "pc/index.html";
        fullLabel = t("pc");
        shortLabel = t("pc_short");
      } else if (viewKey === "ifc") {
        href = root + "ifc.html" + lvlQuery;
        fullLabel = t("ifc");
        shortLabel = t("ifc_short");
      } else {
        continue;
      }

      const seg = document.createElement("a");
      seg.className = "reg-segment" + (activeView === viewKey ? " is-active" : "");
      seg.dataset.key = viewKey;
      if (activeView === viewKey) {
        seg.setAttribute("aria-current", "page");
      }
      seg.href = href;

      const spanFull = document.createElement("span");
      spanFull.className = "reg-full";
      spanFull.textContent = fullLabel;

      const spanShort = document.createElement("span");
      spanShort.className = "reg-short";
      spanShort.textContent = shortLabel;

      seg.appendChild(spanFull);
      seg.appendChild(spanShort);
      segments.appendChild(seg);
    }
    nav.appendChild(segments);

    // Project name on the right
    const titleDiv = document.createElement("div");
    titleDiv.className = "reg-project-name";
    const projName = (project && project.name) || "";
    titleDiv.textContent = projName;
    titleDiv.title = projName;
    nav.appendChild(titleDiv);

    // Insert as FIRST child of <body>
    if (document.body.firstChild) {
      document.body.insertBefore(nav, document.body.firstChild);
    } else {
      document.body.appendChild(nav);
    }

    updateRegHeight();
    window.addEventListener("resize", updateRegHeight);
    window.addEventListener("orientationchange", function () {
      setTimeout(updateRegHeight, 150);
    });

    // Notify window resize after DOM layout
    requestAnimationFrame(function () {
      updateRegHeight();
      window.dispatchEvent(new Event("resize"));
    });
  }

  function run() {
    const projectUrl = root + "data/project.json";

    fetch(projectUrl, { cache: "no-store" })
      .then(function (res) {
        if (!res.ok) throw new Error("HTTP " + res.status);
        return res.json();
      })
      .then(function (project) {
        const lang = (project && project.lang === "en") ? "en" : "de";
        const t = tFactory(lang);

        if (!currentLevel) {
          if (project && project.default_level) {
            currentLevel = project.default_level;
          } else if (project && Array.isArray(project.levels) && project.levels[0]) {
            currentLevel = project.levels[0].key;
          }
          if (currentLevel) setStoredLevel(currentLevel);
        }

        buildBar(project, lang, t);

        window.SAMShare = {
          project: project,
          lang: lang,
          t: t,
          root: root,
          level: currentLevel,
          setLevel: function (lvl) {
            currentLevel = lvl;
            window.SAMShare.level = lvl;
            updateSegmentLinks(lvl);
          }
        };

        window.dispatchEvent(new CustomEvent("samshare:project", { detail: project }));
      })
      .catch(function () {
        const fallbackProject = {
          name: "",
          lang: "de",
          views: ["dxf", "pc", "ifc"]
        };
        const lang = "de";
        const t = tFactory(lang);

        buildBar(fallbackProject, lang, t);

        window.SAMShare = {
          project: fallbackProject,
          lang: lang,
          t: t,
          root: root,
          level: currentLevel,
          setLevel: function (lvl) {
            currentLevel = lvl;
            window.SAMShare.level = lvl;
            updateSegmentLinks(lvl);
          }
        };

        window.dispatchEvent(new CustomEvent("samshare:project", { detail: fallbackProject }));
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", run);
  } else {
    run();
  }
})();
