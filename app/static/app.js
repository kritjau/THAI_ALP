function formatTime(ts) {
  return new Date(ts * 1000).toLocaleTimeString();
}

// Matches the named buckets in app/color.py -- purely a display swatch.
const COLOR_SWATCHES = {
  white: "#f2f2f2",
  silver: "#c9c9c9",
  gray: "#8a8a8a",
  black: "#232323",
  red: "#e5484d",
  orange: "#f7913d",
  yellow: "#eab308",
  green: "#30a46c",
  blue: "#3b82f6",
  brown: "#92603d",
};

function colorSwatch(color) {
  if (!color || !(color in COLOR_SWATCHES)) {
    return `<span class="color-chip"><span class="swatch swatch-unknown"></span>${color || "unknown"}</span>`;
  }
  return `<span class="color-chip"><span class="swatch" style="background:${COLOR_SWATCHES[color]}"></span>${color}</span>`;
}

function setCount(id, n) {
  const el = document.getElementById(id);
  if (el) el.textContent = n;
}

const rowsById = new Map();
const MAX_ROWS = 10;

function registeredBadge(det) {
  return det.registered ? ' <span class="registered-badge">GATE</span>' : "";
}

function cameraTag(det) {
  // Events from before multi-camera support (or a single-camera setup)
  // don't carry camera_name -- falls back to the only camera such a setup
  // ever had, matching db.py's recent_detections() fallback.
  return det.camera_name || "Camera 1";
}

// Capitalizes the vehicle_type label from app/vehicle_detector.py
// ("motorcycle" -> "Motorcycle"); no type (older rows, or no vehicle body
// found for this plate) shows as "?" rather than blank.
function vehicleTypeLabel(det) {
  if (!det.vehicle_type) return "?";
  return det.vehicle_type.charAt(0).toUpperCase() + det.vehicle_type.slice(1);
}

function upsertRow(body, det) {
  const cells = `
    <td class="thumb-cell">${det.image ? `<img src="${det.image}" alt="captured plate" />` : ""}</td>
    <td>${formatTime(det.timestamp)}</td>
    <td class="camera-tag">${cameraTag(det)}</td>
    <td class="plate">${det.plate_text}${registeredBadge(det)}</td>
    <td>${Math.round(det.confidence * 100)}%</td>
    <td>${colorSwatch(det.color)}</td>
    <td>${vehicleTypeLabel(det)}</td>
  `;

  const existing = rowsById.get(det.id);
  if (existing) {
    existing.innerHTML = cells;
    body.prepend(existing); // re-reads move back to the top too, showing latest activity
  } else {
    const row = document.createElement("tr");
    row.innerHTML = cells;
    rowsById.set(det.id, row);
    body.prepend(row);

    while (body.rows.length > MAX_ROWS) {
      const last = body.rows[body.rows.length - 1];
      for (const [id, el] of rowsById) {
        if (el === last) {
          rowsById.delete(id);
          break;
        }
      }
      body.deleteRow(body.rows.length - 1);
    }
  }

  document.getElementById("detections-empty").style.display = "none";
  setCount("detections-count", rowsById.size);
}

async function loadHistory() {
  const body = document.getElementById("detections-body");
  const res = await fetch(`/api/detections?limit=${MAX_ROWS}`);
  const rows = await res.json();
  for (const det of rows.slice().reverse()) {
    upsertRow(body, det);
  }
}

function setConnectionStatus(connected) {
  const el = document.getElementById("connection-status");
  const label = el.querySelector(".status-label");
  el.classList.toggle("status-connected", connected);
  el.classList.toggle("status-disconnected", !connected);
  label.textContent = connected ? "Connected" : "Reconnecting…";
}

function connectWebSocket() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/detections`);
  const body = document.getElementById("detections-body");
  ws.onopen = () => setConnectionStatus(true);
  ws.onmessage = (event) => {
    const det = JSON.parse(event.data);
    upsertRow(body, det);
  };
  ws.onclose = () => {
    setConnectionStatus(false);
    setTimeout(connectWebSocket, 2000);
  };
}

function watchImageStream(img, panel) {
  // The video is a plain <img> pulling a multipart/x-mixed-replace stream,
  // which browsers don't auto-retry -- unlike the WebSocket above, a dropped
  // or silently stalled connection here just stays blank forever without
  // this. `load` fires on every new frame, so a long gap since the last one
  // means the stream died without the browser noticing via `error`.
  //
  // A hidden camera (see setupCameras()) has its `src` deliberately cleared
  // to stop pulling the stream -- skip both the error-triggered reload and
  // the staleness watchdog while hidden, otherwise this would immediately
  // reconnect a feed the viewer just chose to turn off.
  const baseSrc = img.dataset.baseSrc;
  const isHidden = () => panel.classList.contains("camera-hidden");
  let lastFrameAt = Date.now();

  const reload = () => {
    if (isHidden()) return;
    lastFrameAt = Date.now();
    img.src = baseSrc + "?_=" + Date.now();
  };

  img.addEventListener("load", () => {
    lastFrameAt = Date.now();
  });
  img.addEventListener("error", () => setTimeout(reload, 2000));

  setInterval(() => {
    if (isHidden()) {
      lastFrameAt = Date.now(); // don't let the gap build up while intentionally off
      return;
    }
    if (Date.now() - lastFrameAt > 8000) {
      reload();
    }
  }, 3000);
}

// Which cameras a viewer has chosen to hide -- per-browser (localStorage),
// not sent to the server: all cameras keep detecting/recording regardless,
// this only controls what this viewer's dashboard displays.
const HIDDEN_CAMERAS_KEY = "alpr_hidden_cameras";

function loadHiddenCameras() {
  try {
    return new Set(JSON.parse(localStorage.getItem(HIDDEN_CAMERAS_KEY) || "[]"));
  } catch (err) {
    return new Set();
  }
}

function saveHiddenCameras(hidden) {
  try {
    localStorage.setItem(HIDDEN_CAMERAS_KEY, JSON.stringify([...hidden]));
  } catch (err) {
    // private browsing / storage disabled -- toggle still works for this
    // page load, it just won't be remembered next visit
  }
}

function updateNoCamerasMessage(column) {
  let msg = column.querySelector(".no-cameras-message");
  const anyVisible = column.querySelector(".video-panel:not(.camera-hidden)");
  if (anyVisible) {
    if (msg) msg.remove();
    return;
  }
  if (!msg) {
    msg = document.createElement("p");
    msg.className = "empty-state no-cameras-message";
    msg.textContent = "All cameras hidden -- use the toggles above to show one.";
    column.appendChild(msg);
  }
}

async function setupCameras() {
  const column = document.getElementById("video-column");
  let fetched = [];
  try {
    const res = await fetch("/api/cameras");
    if (res.ok) fetched = await res.json();
  } catch (err) {
    // fall through to the single-camera fallback below
  }

  // A camera marked CAMERA_PAGE_N=admin (see app/config.py) exists to
  // trigger the gate, not to be eyeballed here for detection quality --
  // it's shown on the Registered Plates page instead (see admin.html).
  let cameras = fetched.filter((cam) => (cam.page || "main") !== "admin");

  if (fetched.length === 0) {
    // /api/cameras came back empty or unavailable -- fall back to the
    // original single, unlabeled /video_feed rather than showing nothing.
    cameras = [{ id: null, name: null }];
  } else if (cameras.length === 0) {
    // Cameras exist, they're just all admin-only -- distinct from the
    // toggle-oriented "all hidden" message below, since there's nothing
    // here to un-hide.
    column.innerHTML = '<p class="empty-state">No cameras configured for this dashboard.</p>';
    return;
  }

  column.innerHTML = "";
  // A label/toggle is only useful once there's more than one feed to tell
  // apart or choose between -- the common single-camera case stays exactly
  // as before, with no controls cluttering it.
  const showControls = cameras.length > 1;
  const hidden = loadHiddenCameras();

  const toggles = document.createElement("div");
  toggles.className = "camera-toggles";
  if (showControls) column.appendChild(toggles);

  // Two columns once there's more than one camera on this page (a 2x2 grid
  // at 4 cameras) -- a single camera stays full-width, matching how it
  // looked before multi-camera support existed.
  const grid = document.createElement("div");
  grid.className = "video-grid" + (showControls ? "" : " single-camera");
  column.appendChild(grid);

  for (const cam of cameras) {
    const camKey = cam.id ?? "default";
    const src = cam.id ? `/video_feed/${cam.id}` : "/video_feed";
    const isHidden = showControls && hidden.has(camKey);

    const panel = document.createElement("section");
    panel.className = "panel video-panel" + (isHidden ? " camera-hidden" : "");
    panel.innerHTML = `
      ${showControls ? `<span class="camera-label">${cam.name}</span>` : ""}
      <img data-base-src="${src}" alt="${cam.name || "live camera feed"}" />
      <span class="live-badge"><span class="live-dot"></span>LIVE</span>
    `;
    grid.appendChild(panel);

    const img = panel.querySelector("img");
    if (!isHidden) img.src = src;
    watchImageStream(img, panel);

    if (showControls) {
      const label = document.createElement("label");
      label.className = "camera-toggle" + (isHidden ? "" : " active");
      label.innerHTML = `<input type="checkbox" ${isHidden ? "" : "checked"} /> ${cam.name}`;
      label.querySelector("input").addEventListener("change", (e) => {
        const show = e.target.checked;
        panel.classList.toggle("camera-hidden", !show);
        label.classList.toggle("active", show);
        img.src = show ? img.dataset.baseSrc + "?_=" + Date.now() : "";
        const hiddenNow = loadHiddenCameras();
        if (show) hiddenNow.delete(camKey);
        else hiddenNow.add(camKey);
        saveHiddenCameras(hiddenNow);
        updateNoCamerasMessage(column);
      });
      toggles.appendChild(label);
    }
  }

  updateNoCamerasMessage(column);
}

// Fixed display order so segments don't reshuffle as counts change; anything
// outside these four (there shouldn't be, since vehicle_detector.py only
// ever labels one of these) still renders, just appended after with a
// neutral fallback color.
const VEHICLE_TYPE_ORDER = ["car", "motorcycle", "bus", "truck"];
const VEHICLE_TYPE_LABELS = { car: "Car", motorcycle: "Motorcycle", bus: "Bus", truck: "Truck", unknown: "Unknown" };
// One fixed hue per type (never reassigned/cycled) from the dark-mode
// categorical palette, in the order that keeps every pair distinguishable
// going around the ring -- including car/truck, which sit next to each
// other where the ring closes and wouldn't be checked by a plain
// left-to-right "adjacent" list. Two of these (bus, truck) read below ideal
// contrast against the panel background on their own -- the legend's text
// label is what actually carries identity for those, not the swatch alone.
const VEHICLE_TYPE_COLORS = { car: "#3987e5", motorcycle: "#d95926", bus: "#199e70", truck: "#c98500" };
const DONUT_FALLBACK_COLOR = "#8b93a7";

function buildDonutSvg(counts, presentKeys, total) {
  const R = 46, CX = 60, CY = 60, STROKE = 18, GAP = 3;
  const CIRC = 2 * Math.PI * R;
  let offset = 0;
  let segments = `<circle cx="${CX}" cy="${CY}" r="${R}" fill="none" stroke="var(--border)" stroke-width="${STROKE}"></circle>`;
  for (const key of presentKeys) {
    const len = (counts[key] / total) * CIRC;
    const dash = Math.max(len - GAP, 0.001); // small gap between segments, like a stacked bar
    segments += `
      <circle cx="${CX}" cy="${CY}" r="${R}" fill="none"
        stroke="${VEHICLE_TYPE_COLORS[key] || DONUT_FALLBACK_COLOR}" stroke-width="${STROKE}"
        stroke-dasharray="${dash} ${CIRC - dash}" stroke-dashoffset="${-offset}"
        transform="rotate(-90 ${CX} ${CY})"></circle>`;
    offset += len;
  }
  return `<svg viewBox="0 0 120 120" class="donut-svg" role="img" aria-label="Vehicle type breakdown">${segments}</svg>`;
}

function buildVehicleTypeBlock(counts) {
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  if (total === 0) {
    return '<p class="empty-state">No detections yet.</p>';
  }

  const keys = [...VEHICLE_TYPE_ORDER, ...Object.keys(counts).filter((k) => !VEHICLE_TYPE_ORDER.includes(k))];
  const present = keys.filter((k) => counts[k]);

  const legendRows = present
    .map((k) => {
      const pct = Math.round((counts[k] / total) * 100);
      return `
        <div class="legend-row">
          <span class="legend-swatch" style="background:${VEHICLE_TYPE_COLORS[k] || DONUT_FALLBACK_COLOR}"></span>
          <span class="legend-label">${VEHICLE_TYPE_LABELS[k] || k}</span>
          <span class="legend-value">${counts[k]} <span class="legend-pct">(${pct}%)</span></span>
        </div>`;
    })
    .join("");

  return `
    <div class="donut-wrap">
      ${buildDonutSvg(counts, present, total)}
      <div class="donut-total">
        <span class="donut-total-value">${total}</span>
        <span class="donut-total-label">Total</span>
      </div>
    </div>
    <div class="legend">${legendRows}</div>
  `;
}

function rejectBar(pct, label, extraClass = "") {
  return `
    <div class="reject-bar ${extraClass}" role="img" aria-label="${label}">
      <div class="reject-bar-fill" style="transform:scaleX(${pct / 100})"></div>
    </div>
  `;
}

// How many OCR reads made it past the plate-shape check (app/plate_format.py)
// vs. got rejected as implausible -- the real-world signal for whether
// normalization/recognition tuning (upscale height, character voting) is
// actually helping, without needing to grep the server log for "rejected".
// `byCamera` is /api/stats's rejections_by_camera -- only "main"-page
// cameras count toward the overall figure here, same filter setupCameras()
// applies to the video grid: a dedicated gate camera's read quality isn't
// what this monitoring page is checking.
function buildRejectionBlock(byCamera) {
  const cams = (byCamera || []).filter((c) => (c.page || "main") === "main");
  const totals = cams.reduce(
    (acc, c) => ({
      accepted: acc.accepted + (c.accepted || 0),
      rejected: acc.rejected + (c.rejected || 0),
    }),
    { accepted: 0, rejected: 0 }
  );
  const total = totals.accepted + totals.rejected;
  if (total === 0) {
    return '<p class="empty-state">No reads yet.</p>';
  }
  const pct = Math.round((totals.accepted / total) * 100);
  // Only cameras that have actually read something -- keeps this count
  // consistent with the rows shown below it (a freshly-added camera with
  // zero reads yet would otherwise inflate "N cameras" past what's listed).
  const camsWithReads = cams.filter((c) => (c.accepted || 0) + (c.rejected || 0) > 0);

  const perCameraRows = camsWithReads
    .map((c) => {
      const camTotal = (c.accepted || 0) + (c.rejected || 0);
      const camPct = Math.round(((c.accepted || 0) / camTotal) * 100);
      return `
        <div class="camera-reject-row">
          <span class="camera-reject-name">${c.name}</span>
          ${rejectBar(camPct, `${c.name}: ${camPct}% of reads accepted`, "reject-bar-sm")}
          <span class="camera-reject-pct">${camPct}%</span>
        </div>`;
    })
    .join("");

  return `
    <div class="reject-stat">
      ${rejectBar(pct, `${pct}% of reads accepted overall`)}
      <span class="reject-stat-text">
        <strong>${pct}%</strong> accepted overall
        <span class="reject-stat-detail">(${totals.accepted} of ${total} reads, ${camsWithReads.length} camera${camsWithReads.length === 1 ? "" : "s"})</span>
      </span>
      ${perCameraRows ? `<div class="camera-reject-list">${perCameraRows}</div>` : ""}
    </div>
  `;
}

// Local-calendar-day keys (YYYY-MM-DD), oldest to newest, matching how the
// backend buckets vehicle_types_daily (app/db.py's date(...,'localtime')
// and app_live/pipeline.py's time.localtime()) -- note this is the
// *viewer's* local day, so a viewer in a different timezone than the
// server can see a day boundary land a beat earlier/later than the server
// did; not worth reconciling for a single-developer dashboard.
function lastNDayKeys(n) {
  const keys = [];
  const now = new Date();
  for (let i = n - 1; i >= 0; i--) {
    const d = new Date(now.getFullYear(), now.getMonth(), now.getDate() - i);
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, "0");
    const day = String(d.getDate()).padStart(2, "0");
    keys.push(`${y}-${m}-${day}`);
  }
  return keys;
}

function dayLabel(key) {
  const [y, m, d] = key.split("-").map(Number);
  return new Date(y, m - 1, d).toLocaleDateString(undefined, { weekday: "short" });
}

// A small multi-line chart, one line per vehicle type, over the last N
// days -- colors match the donut/legend above it, so no separate legend is
// drawn here.
function buildVehicleTrendChart(daily, days = 7) {
  const dayKeys = lastNDayKeys(days);
  const hasAny = dayKeys.some((k) => daily[k] && Object.values(daily[k]).some((v) => v > 0));
  if (!hasAny) {
    return '<p class="empty-state">Not enough history yet.</p>';
  }

  const types = VEHICLE_TYPE_ORDER.filter((t) => dayKeys.some((k) => (daily[k] || {})[t]));
  const W = 280, H = 100, PAD_X = 10, PAD_TOP = 10, PAD_BOTTOM = 20;
  const plotH = H - PAD_TOP - PAD_BOTTOM;
  const stepX = dayKeys.length > 1 ? (W - PAD_X * 2) / (dayKeys.length - 1) : 0;
  const maxCount = Math.max(1, ...dayKeys.flatMap((k) => types.map((t) => (daily[k] || {})[t] || 0)));
  const x = (i) => PAD_X + i * stepX;
  const y = (v) => H - PAD_BOTTOM - (v / maxCount) * plotH;

  const lines = types
    .map((t) => {
      const values = dayKeys.map((k) => (daily[k] || {})[t] || 0);
      const points = values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
      const dots = values
        .map((v, i) => `<circle cx="${x(i).toFixed(1)}" cy="${y(v).toFixed(1)}" r="2.2" fill="${VEHICLE_TYPE_COLORS[t]}"></circle>`)
        .join("");
      return `<polyline points="${points}" fill="none" stroke="${VEHICLE_TYPE_COLORS[t]}" stroke-width="1.75"></polyline>${dots}`;
    })
    .join("");

  const labels = dayKeys
    .map((k, i) => `<text x="${x(i).toFixed(1)}" y="${H - 5}" font-size="8" fill="var(--muted)" text-anchor="middle">${dayLabel(k)}</text>`)
    .join("");

  return `
    <svg viewBox="0 0 ${W} ${H}" class="trend-svg" role="img" aria-label="Vehicle type counts, last ${days} days">
      ${lines}
      ${labels}
    </svg>
  `;
}

async function loadStats() {
  const vehicleTypesBody = document.getElementById("vehicle-types-body");
  const plateReadsBody = document.getElementById("plate-reads-body");
  if (!vehicleTypesBody || !plateReadsBody) return; // this page has no stats panels
  let data;
  try {
    const res = await fetch("/api/stats");
    if (!res.ok) return;
    data = await res.json();
  } catch (err) {
    return; // leave whatever was last shown rather than blanking it out
  }

  vehicleTypesBody.innerHTML = `
    <div class="stat-block-row">${buildVehicleTypeBlock(data.vehicle_types || {})}</div>
    <div class="stat-block">
      <span class="stat-block-label">Last 7 Days</span>
      ${buildVehicleTrendChart(data.vehicle_types_daily || {})}
    </div>
  `;
  plateReadsBody.innerHTML = buildRejectionBlock(data.rejections_by_camera || []);
}

function revealAdminLinkIfAvailable() {
  // Shared by app/ (no gate feature -- doesn't have this route) and
  // app_live/ (does) -- only show the link where it'll actually work.
  fetch("/api/registered-plates")
    .then((res) => {
      if (res.ok) document.getElementById("admin-link").style.display = "inline";
    })
    .catch(() => {});
}

loadHistory();
connectWebSocket();
setupCameras();
revealAdminLinkIfAvailable();
loadStats();
setInterval(loadStats, 5000);
