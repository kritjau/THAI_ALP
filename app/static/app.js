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

function upsertRow(body, det) {
  const cells = `
    <td>${formatTime(det.timestamp)}</td>
    <td class="plate">${det.plate_text}</td>
    <td>${Math.round(det.confidence * 100)}%</td>
    <td>${colorSwatch(det.color)}</td>
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

    while (body.rows.length > 200) {
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
const MAX_CAPTURES = 30;

function upsertCapture(grid, det) {
  // Only detections that actually carry a captured crop belong here -- the
  // detections table still lists everything, this is just the visual gallery.
  if (!det.image) {
    return;
  }

  const inner = `
    <img src="${det.image}" alt="captured plate" />
    <div class="capture-caption">
      <span class="plate">${det.plate_text}</span>
      <span class="capture-meta">
        ${formatTime(det.timestamp)}
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
  const res = await fetch("/api/detections?limit=50");
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

function watchVideoFeed() {
  // The video is a plain <img> pulling a multipart/x-mixed-replace stream,
  // which browsers don't auto-retry -- unlike the WebSocket above, a dropped
  // or silently stalled connection here just stays blank forever without
  // this. `load` fires on every new frame, so a long gap since the last one
  // means the stream died without the browser noticing via `error`.
  const img = document.getElementById("video-feed");
  let lastFrameAt = Date.now();

  const reload = () => {
    lastFrameAt = Date.now();
    img.src = "/video_feed?_=" + Date.now();
  };

  img.addEventListener("load", () => {
    lastFrameAt = Date.now();
  });
  img.addEventListener("error", () => setTimeout(reload, 2000));

  setInterval(() => {
    if (Date.now() - lastFrameAt > 8000) {
      reload();
    }
  }, 3000);
}

loadHistory();
connectWebSocket();
watchVideoFeed();
