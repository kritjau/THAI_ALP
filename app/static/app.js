function formatTime(ts) {
  return new Date(ts * 1000).toLocaleTimeString();
}

const rowsById = new Map();

function upsertRow(body, det) {
  const cells = `
    <td>${formatTime(det.timestamp)}</td>
    <td class="plate">${det.plate_text}</td>
    <td>${Math.round(det.confidence * 100)}%</td>
    <td>${det.vehicle_color || "-"}</td>
  `;

  const existing = rowsById.get(det.id);
  if (existing) {
    existing.innerHTML = cells;
    body.prepend(existing); // re-reads move back to the top too, showing latest activity
    return;
  }

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

async function loadHistory() {
  const body = document.getElementById("detections-body");
  const res = await fetch("/api/detections?limit=50");
  const rows = await res.json();
  for (const det of rows.slice().reverse()) {
    upsertRow(body, det);
  }
}

function connectWebSocket() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/detections`);
  const body = document.getElementById("detections-body");
  ws.onmessage = (event) => {
    upsertRow(body, JSON.parse(event.data));
  };
  ws.onclose = () => setTimeout(connectWebSocket, 2000);
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
