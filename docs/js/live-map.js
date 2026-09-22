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

  function turboRGB(t) {
    t = Math.max(0, Math.min(1, t));
    const r = Math.round(255 * Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 3))));
    const g = Math.round(255 * Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 2))));
    const b = Math.round(255 * Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 1))));
    return [r, g, b];
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

    const { vmin, vmax } = finiteStats(grid);
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
          img.data[p] = 15;
          img.data[p + 1] = 23;
          img.data[p + 2] = 42;
          img.data[p + 3] = 255;
          continue;
        }
        const [r, g, b] = turboRGB((v - vmin) / span);
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

    ctx.fillStyle = "#0b1220";
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
    const canvas = document.getElementById("live-canvas");
    if (!canvas) return;

    let mode = "surface";
    let fieldId = "sst";
    let payload = null;
    let busy = false;
    let cssSize = { w: 960, h: 480 };

    const statusText = document.getElementById("live-status-text");
    const updatedEl = document.getElementById("live-updated");
    const dateEl = document.getElementById("live-date");
    const titleEl = document.getElementById("live-viewer-title");
    const gridInfo = document.getElementById("live-grid-info");
    const legendMin = document.getElementById("live-legend-min");
    const legendMax = document.getElementById("live-legend-max");
    const legendUnit = document.getElementById("live-legend-unit");
    const intervalSec = document.getElementById("live-interval-sec");

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

    function fitCanvas(grid) {
      const wrap = canvas.parentElement;
      if (!wrap) return;

      const cssW = Math.max(320, Math.floor(wrap.clientWidth));
      const H = grid?.length || 100;
      const W = grid?.[0]?.length || 240;
      const aspect = geoAspect(payload?.lat, payload?.lon, H, W);

      // Cap at 3x for Retina/HiDPI; render map buffer at this density
      const dpr = Math.min(Math.max(window.devicePixelRatio || 1, 1), 3);
      const plotW = Math.max(1, cssW - PAD.L - PAD.R);
      const plotH = plotW / aspect;
      const cssH = Math.round(plotH + PAD.T + PAD.B);
      const bw = Math.max(1, Math.round(cssW * dpr));
      const bh = Math.max(1, Math.round(cssH * dpr));

      cssSize = { w: cssW, h: cssH, dpr };
      if (canvas.width !== bw || canvas.height !== bh) {
        canvas.width = bw;
        canvas.height = bh;
      }
      canvas.style.width = cssW + "px";
      canvas.style.height = cssH + "px";
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

      fitCanvas(grid);

      const ctx = canvas.getContext("2d");
      const dpr = cssSize.dpr || 1;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      if (!grid) {
        ctx.fillStyle = "#0b1220";
        ctx.fillRect(0, 0, cssSize.w, cssSize.h);
        ctx.fillStyle = "#94a3b8";
        ctx.font = "14px Inter, system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(
          mode === "pred" && payload && !Object.keys(payload.pred || {}).length
            ? "No ViT weights loaded on server"
            : "Waiting for data…",
          cssSize.w / 2,
          cssSize.h / 2
        );
        return;
      }

      const lat0 = payload.lat[0];
      const lat1 = payload.lat[payload.lat.length - 1];
      const lon0 = payload.lon[0];
      const lon1 = payload.lon[payload.lon.length - 1];
      gridInfo.textContent = `${grid.length} × ${grid[0].length} · ${Math.min(lat0, lat1).toFixed(1)}–${Math.max(lat0, lat1).toFixed(1)}°N, ${lon0.toFixed(1)}–${lon1.toFixed(1)}°E`;

      paintColormap(ctx, cssSize.w, cssSize.h, grid, {
        dpr,
        lat: payload.lat,
        lon: payload.lon,
        onRange(vmin, vmax) {
          legendMin.textContent = vmin.toFixed(2);
          legendMax.textContent = vmax.toFixed(2);
        },
      });
    }

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
        model: "demo",
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

    document.getElementById("live-refresh-btn")?.addEventListener("click", () => fetchLive());

    let resizeTimer = null;
    window.addEventListener("resize", () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => redraw(), 100);
    });

    const wrap = canvas.parentElement;
    if (wrap && typeof ResizeObserver !== "undefined") {
      new ResizeObserver(() => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => redraw(), 50);
      }).observe(wrap);
    }

    setInterval(fetchLive, POLL_MS);
    fetchLive();
  });
})();
