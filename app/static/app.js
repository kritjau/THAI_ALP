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
const MAX_ROWS = 12;

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

const capturesById = new Map();
const MAX_CAPTURES = 6;

function upsertCapture(grid, det) {
  // Only detections that actually carry a captured crop belong here -- the
  // detections table still lists everything, this is just the visual gallery.
  if (!det.image) {
    return;
  }

  const inner = `
    <img src="${det.image}" alt="captured plate" />
    <div class="capture-caption">
      <span class="plate">${det.plate_text}${registeredBadge(det)}</span>
      <span class="capture-meta">
        ${formatTime(det.timestamp)} &middot; ${cameraTag(det)}
        ${det.vehicle_type ? `&middot; ${vehicleTypeLabel(det)}` : ""}
        ${det.color ? `<span class="swatch" style="background:${COLOR_SWATCHES[det.color] || "#666"}"></span>` : ""}
      </span>
    </div>
  `;

  const existing = capturesById.get(det.id);
  if (existing) {
    existing.innerHTML = inner;
    grid.prepend(existing);
  } else {
    const empty = grid.querySelector(".empty-state");
    if (empty) empty.remove();

    const card = document.createElement("div");
    card.className = "capture-card";
    card.innerHTML = inner;
    capturesById.set(det.id, card);
    grid.prepend(card);

    while (grid.children.length > MAX_CAPTURES) {
      const last = grid.lastElementChild;
      for (const [id, el] of capturesById) {
        if (el === last) {
          capturesById.delete(id);
          break;
        }
      }
      grid.removeChild(last);
    }
  }

  setCount("captures-count", capturesById.size);
}

async function loadHistory() {
  const body = document.getElementById("detections-body");
  const grid = document.getElementById("captures-grid");
  const res = await fetch("/api/detections?limit=12");
  const rows = await res.json();
  for (const det of rows.slice().reverse()) {
    upsertRow(body, det);
    upsertCapture(grid, det);
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
  const grid = document.getElementById("captures-grid");
  ws.onopen = () => setConnectionStatus(true);
  ws.onmessage = (event) => {
    const det = JSON.parse(event.data);
    upsertRow(body, det);
    upsertCapture(grid, det);
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
  let cameras = [];
  try {
    const res = await fetch("/api/cameras");
    if (res.ok) cameras = await res.json();
  } catch (err) {
    // fall through to the single-camera fallback below
  }
  if (!cameras || cameras.length === 0) {
    // /api/cameras came back empty or unavailable -- fall back to the
    // original single, unlabeled /video_feed rather than showing nothing.
    cameras = [{ id: null, name: null }];
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
    column.appendChild(panel);

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
