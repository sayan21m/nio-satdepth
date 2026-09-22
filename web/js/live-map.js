/**
 * Live colormap maps — filled pcolormesh-style canvas for surface + predicted θ.
 * Expects docs/live_server.py on the same origin (http://127.0.0.1:8765).
 */
(function () {
  const POLL_MS = 5000;
  const PAD = { L: 56, R: 18, T: 18, B: 42 };
  const SURFACE_FIELDS = [
    { id: "sst", label: "SST", unit: "°C" },
    { id: "sss", label: "SSS", unit: "psu" },
    { id: "sla", label: "SLA", unit: "m" },
    { id: "adt", label: "ADT", unit: "m" },
    { id: "uo", label: "uo", unit: "m/s" },
    { id: "vo", label: "vo", unit: "m/s" },
    { id: "u10", label: "u10", unit: "m/s" },
    { id: "v10", label: "v10", unit: "m/s" },
  ];
  const DEPTH_FIELDS = [
    { id: "0m", label: "0 m", unit: "°C" },
    { id: "50m", label: "50 m", unit: "°C" },
    { id: "100m", label: "100 m", unit: "°C" },
    { id: "150m", label: "150 m", unit: "°C" },
  ];

  // Matplotlib turbo (Google) control points
  const TURBO_STOPS = [
    [0.0, 48, 18, 59],
    [0.1, 70, 69, 172],
    [0.2, 67, 135, 247],
    [0.35, 42, 176, 203],
    [0.5, 34, 199, 112],
    [0.65, 164, 218, 54],
    [0.8, 254, 185, 39],
    [0.9, 251, 110, 32],
    [1.0, 122, 4, 3],
  ];

  function lerpRgb(stops, t) {
    t = Math.max(0, Math.min(1, t));
    let i = 0;
    while (i < stops.length - 2 && t > stops[i + 1][0]) i++;
    const a = stops[i];
    const b = stops[i + 1];
    const u = (t - a[0]) / Math.max(b[0] - a[0], 1e-9);
    return [
      Math.round(a[1] + (b[1] - a[1]) * u),
      Math.round(a[2] + (b[2] - a[2]) * u),
      Math.round(a[3] + (b[3] - a[3]) * u),
    ];
  }

  function turboRGB(t) {
    return lerpRgb(TURBO_STOPS, t);
  }

  function rdBuRGB(t) {
    // matplotlib RdBu_r: blue (low) → white → red (high)
    const x = Math.max(0, Math.min(1, t));
    if (x < 0.5) {
      const u = x / 0.5;
      return [
        Math.round(33 + (247 - 33) * u),
        Math.round(102 + (247 - 102) * u),
        Math.round(172 + (247 - 172) * u),
      ];
    }
    const u = (x - 0.5) / 0.5;
    return [
      Math.round(247 + (178 - 247) * u),
      Math.round(247 + (24 - 247) * u),
      Math.round(247 + (43 - 247) * u),
    ];
  }

  function divergingField(id) {
    return id === "sla" || id === "adt" || id === "uo" || id === "vo" || id === "u10" || id === "v10";
  }

  function finiteStats(grid) {
    let vmin = Infinity,
      vmax = -Infinity,
      n = 0;
    for (const row of grid) {
      for (const v of row) {
        if (v == null || Number.isNaN(v)) continue;
        n++;
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
      }
    }
    if (!n) return { vmin: 0, vmax: 1, n: 0 };
    if (vmax - vmin < 1e-9) {
      vmin -= 0.5;
      vmax += 0.5;
    }
    return { vmin, vmax, n };
  }

  function geoAspect(lat, lon, H, W) {
    if (lat?.length > 1 && lon?.length > 1) {
      const dlat = Math.abs(lat[lat.length - 1] - lat[0]);
      const dlon = Math.abs(lon[lon.length - 1] - lon[0]);
      if (dlat > 0 && dlon > 0) return dlon / dlat;
    }
    return W / Math.max(H, 1);
  }

  function sampleBilinear(grid, H, W, x, y) {
    // x,y in source cell coords; north-up: y=0 is north (src row H-1)
    const srcY = H - 1 - y;
    const x0 = Math.floor(x);
    const y0 = Math.floor(srcY);
    const x1 = Math.min(W - 1, x0 + 1);
    const y1 = Math.min(H - 1, Math.max(0, y0 + 1));
    const fx = x - x0;
    const fy = srcY - y0;
    const cx0 = Math.max(0, Math.min(W - 1, x0));
    const cy0 = Math.max(0, Math.min(H - 1, y0));

    const v00 = grid[cy0][cx0];
    const v10 = grid[cy0][x1];
    const v01 = grid[y1][cx0];
    const v11 = grid[y1][x1];
    const miss = (v) => v == null || Number.isNaN(v);
    if (miss(v00) && miss(v10) && miss(v01) && miss(v11)) return NaN;

    const a = miss(v00) ? 0 : v00;
    const b = miss(v10) ? a : v10;
    const c = miss(v01) ? a : v01;
    const d = miss(v11) ? b : v11;
    const w00 = miss(v00) ? 0 : (1 - fx) * (1 - fy);
    const w10 = miss(v10) ? 0 : fx * (1 - fy);
    const w01 = miss(v01) ? 0 : (1 - fx) * fy;
    const w11 = miss(v11) ? 0 : fx * fy;
    const ws = w00 + w10 + w01 + w11;
    if (ws < 1e-9) return NaN;
    return (a * w00 + b * w10 + c * w01 + d * w11) / ws;
  }

  /** Draw filled colormap in CSS-pixel space (ctx already scaled by DPR). */
  function paintColormap(ctx, cssW, cssH, grid, opts = {}) {
    const H = grid.length;
    const W = grid[0]?.length || 0;
    if (!H || !W) return;

    let { vmin, vmax } = finiteStats(grid);
    const diverging = !!opts.diverging;
    if (diverging) {
      const lim = Math.max(Math.abs(vmin), Math.abs(vmax), 1e-6);
      vmin = -lim;
      vmax = lim;
    }
    const { L: padL, R: padR, T: padT, B: padB } = PAD;
    const plotW = cssW - padL - padR;
    const plotH = cssH - padT - padB;
    const dpr = opts.dpr || 1;

    // Render map at device-pixel resolution with bilinear upsampling
    const outW = Math.max(1, Math.round(plotW * dpr));
    const outH = Math.max(1, Math.round(plotH * dpr));
    const img = ctx.createImageData(outW, outH);
    const span = vmax - vmin;

    for (let j = 0; j < outH; j++) {
      const gy = ((j + 0.5) / outH) * (H - 1);
      for (let i = 0; i < outW; i++) {
        const gx = ((i + 0.5) / outW) * (W - 1);
        const v = sampleBilinear(grid, H, W, gx, gy);
        const p = (j * outW + i) * 4;
        if (v == null || Number.isNaN(v)) {
          img.data[p] = 0;
          img.data[p + 1] = 0;
          img.data[p + 2] = 0;
          img.data[p + 3] = 255;
          continue;
        }
        const tn = (v - vmin) / span;
        const [r, g, b] = diverging ? rdBuRGB(tn) : turboRGB(tn);
        img.data[p] = r;
        img.data[p + 1] = g;
        img.data[p + 2] = b;
        img.data[p + 3] = 255;
      }
    }

    const tmp = document.createElement("canvas");
    tmp.width = outW;
    tmp.height = outH;
    tmp.getContext("2d").putImageData(img, 0, 0);

    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, cssW, cssH);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(tmp, padL, padT, plotW, plotH);

    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 1;
    ctx.strokeRect(padL + 0.5, padT + 0.5, plotW - 1, plotH - 1);

    ctx.fillStyle = "#94a3b8";
    ctx.font = "12px Inter, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Longitude (°E)", padL + plotW / 2, cssH - 8);
    ctx.save();
    ctx.translate(16, padT + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Latitude (°N)", 0, 0);
    ctx.restore();

    if (opts.lat && opts.lon) {
      const lat0 = opts.lat[0];
      const lat1 = opts.lat[opts.lat.length - 1];
      const lon0 = opts.lon[0];
      const lon1 = opts.lon[opts.lon.length - 1];
      const south = Math.min(lat0, lat1);
      const north = Math.max(lat0, lat1);
      ctx.textAlign = "left";
      ctx.fillText(north.toFixed(1) + "°N", 4, padT + 12);
      ctx.fillText(south.toFixed(1) + "°N", 4, padT + plotH);
      ctx.textAlign = "center";
      ctx.fillText(lon0.toFixed(1), padL, padT + plotH + 16);
      ctx.fillText(lon1.toFixed(1), padL + plotW, padT + plotH + 16);
    }

    if (opts.onRange) opts.onRange(vmin, vmax);
  }

  function renderRaster(grid, opts = {}) {
    const H = grid.length;
    const W = grid[0]?.length || 0;
    const scale = opts.scale || 8;
    const outW = Math.max(1, W * scale);
    const outH = Math.max(1, H * scale);
    let { vmin, vmax } = finiteStats(grid);
    if (opts.diverging) {
      const lim = Math.max(Math.abs(vmin), Math.abs(vmax), 1e-6);
      vmin = -lim;
      vmax = lim;
    }
    const span = vmax - vmin;
    const canvas = document.createElement("canvas");
    canvas.width = outW;
    canvas.height = outH;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(outW, outH);
    for (let j = 0; j < outH; j++) {
      const gy = ((j + 0.5) / outH) * (H - 1);
      for (let i = 0; i < outW; i++) {
        const gx = ((i + 0.5) / outW) * (W - 1);
        const v = sampleBilinear(grid, H, W, gx, gy);
        const p = (j * outW + i) * 4;
        if (v == null || Number.isNaN(v)) {
          img.data[p] = 0;
          img.data[p + 1] = 0;
          img.data[p + 2] = 0;
          img.data[p + 3] = 0;
          continue;
        }
        const [r, g, b] = opts.diverging ? rdBuRGB((v - vmin) / span) : turboRGB((v - vmin) / span);
        img.data[p] = r;
        img.data[p + 1] = g;
        img.data[p + 2] = b;
        img.data[p + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    if (opts.onRange) opts.onRange(vmin, vmax);
    return canvas.toDataURL("image/png");
  }

  function fieldBounds(lat, lon) {
    const south = Math.min(lat[0], lat[lat.length - 1]);
    const north = Math.max(lat[0], lat[lat.length - 1]);
    const west = Math.min(lon[0], lon[lon.length - 1]);
    const east = Math.max(lon[0], lon[lon.length - 1]);
    const dlat = Math.abs(lat[1] - lat[0]) || 0.25;
    const dlon = Math.abs(lon[1] - lon[0]) || 0.25;
    return [
      [south - dlat / 2, west - dlon / 2],
      [north + dlat / 2, east + dlon / 2],
    ];
  }

  function sampleAtLatLon(grid, lat, lon, qlat, qlon) {
    const H = grid.length;
    const W = grid[0]?.length || 0;
    if (!H || !W) return NaN;
    const lat0 = lat[0];
    const lat1 = lat[lat.length - 1];
    const lon0 = lon[0];
    const lon1 = lon[lon.length - 1];
    const rowFromSouth = ((qlat - lat0) / (lat1 - lat0)) * (H - 1);
    const col = ((qlon - lon0) / (lon1 - lon0)) * (W - 1);
    if (rowFromSouth < -0.5 || rowFromSouth > H - 0.5 || col < -0.5 || col > W - 0.5) return NaN;
    const yFromNorth = H - 1 - rowFromSouth;
    return sampleBilinear(grid, H, W, col, yFromNorth);
  }

  function setActiveTabs(container, activeBtn) {
    container.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    activeBtn.classList.add("active");
  }

  function rebuildFieldTabs(mode, activeId, onPick) {
    const wrap = document.getElementById("live-field-tabs");
    const label = document.getElementById("live-field-label");
    if (!wrap) return;
    const fields = mode === "surface" ? SURFACE_FIELDS : DEPTH_FIELDS;
    if (label) label.textContent = mode === "surface" ? "Surface Channel" : "Depth";
    wrap.innerHTML = "";
    fields.forEach((f, i) => {
      const btn = document.createElement("button");
      btn.className = "tab-btn" + ((activeId ? f.id === activeId : i === 0) ? " active" : "");
      btn.dataset.field = f.id;
      btn.textContent = f.label;
      btn.addEventListener("click", () => {
        setActiveTabs(wrap, btn);
        onPick(f.id);
      });
      wrap.appendChild(btn);
    });
    return fields.find((f) => f.id === activeId) || fields[0];
  }

  document.addEventListener("DOMContentLoaded", () => {
    const mapEl = document.getElementById("live-map");
    if (!mapEl || typeof L === "undefined") return;

    let mode = "surface";
    let fieldId = "sst";
    let payload = null;
    let busy = false;
    let overlay = null;
    let fitted = false;

    const hoverEl = document.getElementById("live-hover");
    const statusText = document.getElementById("live-status-text");
    const updatedEl = document.getElementById("live-updated");
    const dateEl = document.getElementById("live-date");
    const titleEl = document.getElementById("live-viewer-title");
    const gridInfo = document.getElementById("live-grid-info");
    const legendMin = document.getElementById("live-legend-min");
    const legendMax = document.getElementById("live-legend-max");
    const legendUnit = document.getElementById("live-legend-unit");
    const legendBar = document.querySelector(".live-legend-bar");
    const intervalSec = document.getElementById("live-interval-sec");

    const map = L.map(mapEl, {
      zoomControl: true,
      minZoom: 3,
      maxZoom: 9,
      worldCopyJump: false,
      attributionControl: true,
    }).setView([17.5, 75], 5);
    window.__liveMap = map;

    L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
      { attribution: "Tiles &copy; Esri", maxZoom: 16 }
    ).addTo(map);

    L.control.scale({ metric: true, imperial: false }).addTo(map);

    const resetBtn = document.createElement("button");
    resetBtn.type = "button";
    resetBtn.className = "tab-btn live-map-reset";
    resetBtn.textContent = "Reset view";
    mapEl.appendChild(resetBtn);

    mapEl.addEventListener("wheel", (e) => e.stopPropagation(), { capture: true });
    mapEl.addEventListener("touchmove", (e) => e.stopPropagation(), { capture: true });

    function setStatus(msg) {
      if (statusText) statusText.textContent = msg;
    }

    function currentMeta() {
      if (mode === "surface") {
        return SURFACE_FIELDS.find((f) => f.id === fieldId) || SURFACE_FIELDS[0];
      }
      return DEPTH_FIELDS.find((f) => f.id === fieldId) || DEPTH_FIELDS[0];
    }

    function pickGrid() {
      if (!payload) return null;
      if (mode === "surface") return payload.surface?.[fieldId] || null;
      if (mode === "pred") return payload.pred?.[fieldId] || null;
      return payload.true?.[fieldId] || null;
    }

    function nioBounds() {
      if (payload?.lat && payload?.lon) return fieldBounds(payload.lat, payload.lon);
      return [[5, 45], [30, 105]];
    }

    function redraw() {
      const grid = pickGrid();
      const meta = currentMeta();
      titleEl.textContent =
        mode === "surface"
          ? meta.label
          : mode === "pred"
            ? `Predicted θ @ ${meta.label}`
            : `True θ @ ${meta.label}`;
      legendUnit.textContent = meta.unit;
      if (legendBar) {
        legendBar.style.background = divergingField(fieldId)
          ? "linear-gradient(90deg, #2166ac, #f7f7f7, #b2182b)"
          : "linear-gradient(90deg, #30123b, #1d4ed8, #22d3ee, #fbbf24, #ef4444, #7a0403)";
      }

      if (!grid || !payload?.lat) {
        if (hoverEl) hoverEl.textContent = "Waiting for data…";
        return;
      }

      const lat0 = payload.lat[0];
      const lat1 = payload.lat[payload.lat.length - 1];
      const lon0 = payload.lon[0];
      const lon1 = payload.lon[payload.lon.length - 1];
      gridInfo.textContent = `${grid.length} × ${grid[0].length} · ${Math.min(lat0, lat1).toFixed(1)}–${Math.max(lat0, lat1).toFixed(1)}°N, ${lon0.toFixed(1)}–${lon1.toFixed(1)}°E`;

      const bounds = fieldBounds(payload.lat, payload.lon);
      const url = renderRaster(grid, {
        scale: 8,
        diverging: divergingField(fieldId),
        onRange(vmin, vmax) {
          legendMin.textContent = vmin.toFixed(2);
          legendMax.textContent = vmax.toFixed(2);
        },
      });

      if (overlay) {
        overlay.setUrl(url);
        overlay.setBounds(bounds);
      } else {
        overlay = L.imageOverlay(url, bounds, { opacity: 0.92, interactive: true }).addTo(map);
      }
      map.invalidateSize();
      if (!fitted) {
        const refit = () => {
          map.invalidateSize();
          if (mapEl.clientHeight < 80) return;
          map.fitBounds(bounds, { padding: [12, 12], maxZoom: 6, animate: false });
          fitted = true;
        };
        refit();
        requestAnimationFrame(refit);
        setTimeout(refit, 250);
      }
    }

    map.on("mousemove", (e) => {
      const grid = pickGrid();
      if (!hoverEl) return;
      if (!grid || !payload?.lat) {
        hoverEl.textContent = "Scroll to zoom · drag to pan";
        return;
      }
      const { lat, lng } = e.latlng;
      const v = sampleAtLatLon(grid, payload.lat, payload.lon, lat, lng);
      const meta = currentMeta();
      const val = v == null || Number.isNaN(v) ? "land / no data" : `${v.toFixed(2)} ${meta.unit}`;
      hoverEl.textContent = `${lat.toFixed(2)}°N  ${lng.toFixed(2)}°E  ·  ${val}`;
    });

    resetBtn.addEventListener("click", () => {
      map.fitBounds(nioBounds(), { padding: [16, 16] });
    });

    let demoPack = null;
    let demoIdx = 0;

    function applyPayload(data) {
      payload = data;
      if (intervalSec && data.advance_sec) intervalSec.textContent = String(data.advance_sec);
      dateEl.textContent = data.date;
      updatedEl.textContent = new Date().toLocaleTimeString();
      setStatus(
        data.model === "demo"
          ? "Demo playback"
          : data.model
            ? `Live · ViT ${data.model}`
            : "Live · surface only"
      );
      redraw();
    }

    async function loadStaticDemo() {
      if (!demoPack) {
        const res = await fetch("/data/live_demo.json", { cache: "no-store" });
        if (!res.ok) throw new Error(`demo HTTP ${res.status}`);
        demoPack = await res.json();
      }
      const days = demoPack.days || [];
      if (!days.length) throw new Error("empty demo");
      const day = days[demoIdx % days.length];
      demoIdx += 1;
      applyPayload({
        ok: true,
        date: day.date,
        model: demoPack.model || "demo",
        lat: demoPack.lat,
        lon: demoPack.lon,
        surface: day.surface,
        pred: day.pred || {},
        true: day.true || {},
        advance_sec: 20,
        n: days.length,
      });
    }

    async function fetchLive() {
      if (busy) return;
      busy = true;
      setStatus("Updating…");
      try {
        const res = await fetch("/api/live", { cache: "no-store" });
        if (res.ok) {
          const data = await res.json();
          if (data.ok) {
            applyPayload(data);
            return;
          }
        }
        await loadStaticDemo();
      } catch (err) {
        console.error(err);
        try {
          await loadStaticDemo();
        } catch (demoErr) {
          console.error(demoErr);
          setStatus("Start live_server.py (port 8765)");
        }
      } finally {
        busy = false;
      }
    }

    rebuildFieldTabs(mode, fieldId, (id) => {
      fieldId = id;
      redraw();
    });

    document.querySelectorAll("#live-mode-tabs .tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        setActiveTabs(document.getElementById("live-mode-tabs"), btn);
        mode = btn.dataset.mode;
        fieldId = mode === "surface" ? "sst" : "100m";
        rebuildFieldTabs(mode, fieldId, (id) => {
          fieldId = id;
          redraw();
        });
        redraw();
      });
    });

    document.querySelectorAll("#live-refresh-btn").forEach((btn) => {
      btn.addEventListener("click", () => fetchLive());
    });

    let resizeTimer = null;
    window.addEventListener("resize", () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        map.invalidateSize();
        redraw();
      }, 100);
    });

    setInterval(fetchLive, POLL_MS);
    fetchLive();
  });
})();
