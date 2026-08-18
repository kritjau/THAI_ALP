function formatTime(ts) {
  return new Date(ts * 1000).toLocaleTimeString();
}

function prependRow(body, det) {
  const row = document.createElement("tr");
  row.innerHTML = `
    <td>${formatTime(det.timestamp)}</td>
    <td class="plate">${det.plate_text}</td>
    <td>${Math.round(det.confidence * 100)}%</td>
  `;
  body.prepend(row);
  while (body.rows.length > 200) {
    body.deleteRow(body.rows.length - 1);
  }
}

async function loadHistory() {
  const body = document.getElementById("detections-body");
  const res = await fetch("/api/detections?limit=50");
  const rows = await res.json();
  for (const det of rows) {
    prependRow(body, det);
  }
}

function connectWebSocket() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/detections`);
  const body = document.getElementById("detections-body");
  ws.onmessage = (event) => {
    prependRow(body, JSON.parse(event.data));
  };
  ws.onclose = () => setTimeout(connectWebSocket, 2000);
}

loadHistory();
connectWebSocket();
