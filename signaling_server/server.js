const WebSocket = require('ws');
const wss = new WebSocket.Server({ port: 8080 });

let sessions = {};  // Store clients by sessionName

wss.on('connection', function connection(ws) {
  ws.on('message', function incoming(message) {
    const { sessionName, signalData } = JSON.parse(message);
    console.log(`Received message from ${sessionName}: ${signalData}`);
    
    // If this session doesn't exist, create it
    if (!sessions[sessionName]) {
      sessions[sessionName] = [];
    }

    // Store the client in this session if we haven't yet
    if (!sessions[sessionName].includes(ws)) {
      sessions[sessionName].push(ws);
    }

    if (signalData === "connect_session") {
        // don't broadcast
        return
    }
    

    // Broadcast the signalData to all other peers in the session
    sessions[sessionName].forEach(function each(client) {
      if (client !== ws && client.readyState === WebSocket.OPEN) {
        client.send(JSON.stringify({ signalData }));
      }
    });
  });
});

console.log("WebSocket signaling server running on ws://localhost:8080");
