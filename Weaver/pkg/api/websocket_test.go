package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

// -----------------------------------------------------------------------------
// Hub Tests
// -----------------------------------------------------------------------------

func TestNewHub(t *testing.T) {
	hub := NewHub()
	if hub == nil {
		t.Fatal("Expected hub to be created")
	}
	if hub.clients == nil {
		t.Error("Expected clients map to be initialized")
	}
	if hub.broadcast == nil {
		t.Error("Expected broadcast channel to be initialized")
	}
	if hub.register == nil {
		t.Error("Expected register channel to be initialized")
	}
	if hub.unregister == nil {
		t.Error("Expected unregister channel to be initialized")
	}
}

func TestHub_ClientCount(t *testing.T) {
	hub := NewHub()
	go hub.Run()
	defer hub.Stop()

	if count := hub.ClientCount(); count != 0 {
		t.Errorf("Expected 0 clients initially, got %d", count)
	}
}

func TestHub_RunAndStop(t *testing.T) {
	hub := NewHub()

	// Run in goroutine
	done := make(chan struct{})
	go func() {
		hub.Run()
		close(done)
	}()

	// Give it time to start
	time.Sleep(10 * time.Millisecond)

	// Stop should cause Run to return
	hub.Stop()

	select {
	case <-done:
		// Success
	case <-time.After(1 * time.Second):
		t.Error("Hub.Run did not stop after Stop was called")
	}
}

// -----------------------------------------------------------------------------
// Client Tests
// -----------------------------------------------------------------------------

func TestNewClient(t *testing.T) {
	hub := NewHub()
	// Note: We're passing nil for conn since we're only testing client creation
	client := NewClient(hub, nil)

	if client == nil {
		t.Fatal("Expected client to be created")
	}
	if client.hub != hub {
		t.Error("Expected client.hub to be set")
	}
	if client.send == nil {
		t.Error("Expected client.send to be initialized")
	}
	if client.subscriptions == nil {
		t.Error("Expected client.subscriptions to be initialized")
	}
}

func TestClient_Subscribe(t *testing.T) {
	hub := NewHub()
	client := NewClient(hub, nil)

	t.Run("subscribe to single channel", func(t *testing.T) {
		client.Subscribe(ChannelMeasurements)

		if !client.IsSubscribed(ChannelMeasurements) {
			t.Error("Expected client to be subscribed to measurements")
		}
	})

	t.Run("subscribe to multiple channels", func(t *testing.T) {
		client.Subscribe(ChannelMessages, ChannelStatus)

		if !client.IsSubscribed(ChannelMessages) {
			t.Error("Expected client to be subscribed to messages")
		}
		if !client.IsSubscribed(ChannelStatus) {
			t.Error("Expected client to be subscribed to status")
		}
	})

	t.Run("not subscribed to unknown channel", func(t *testing.T) {
		if client.IsSubscribed("unknown") {
			t.Error("Expected client to not be subscribed to unknown channel")
		}
	})
}

func TestClient_Unsubscribe(t *testing.T) {
	hub := NewHub()
	client := NewClient(hub, nil)

	client.Subscribe(ChannelMeasurements, ChannelMessages)

	t.Run("unsubscribe from single channel", func(t *testing.T) {
		client.Unsubscribe(ChannelMeasurements)

		if client.IsSubscribed(ChannelMeasurements) {
			t.Error("Expected client to be unsubscribed from measurements")
		}
		if !client.IsSubscribed(ChannelMessages) {
			t.Error("Expected client to still be subscribed to messages")
		}
	})

	t.Run("unsubscribe from nonexistent channel", func(t *testing.T) {
		// Should not panic
		client.Unsubscribe("nonexistent")
	})
}

func TestClient_Subscriptions(t *testing.T) {
	hub := NewHub()
	client := NewClient(hub, nil)

	t.Run("empty subscriptions", func(t *testing.T) {
		subs := client.Subscriptions()
		if len(subs) != 0 {
			t.Errorf("Expected 0 subscriptions, got %d", len(subs))
		}
	})

	t.Run("with subscriptions", func(t *testing.T) {
		client.Subscribe(ChannelMeasurements, ChannelMessages)
		subs := client.Subscriptions()
		if len(subs) != 2 {
			t.Errorf("Expected 2 subscriptions, got %d", len(subs))
		}

		// Check that both channels are in the list
		found := make(map[string]bool)
		for _, s := range subs {
			found[s] = true
		}
		if !found[ChannelMeasurements] {
			t.Error("Expected measurements in subscriptions")
		}
		if !found[ChannelMessages] {
			t.Error("Expected messages in subscriptions")
		}
	})
}

// -----------------------------------------------------------------------------
// WSMessage Tests
// -----------------------------------------------------------------------------

func TestWSMessage_JSON(t *testing.T) {
	t.Run("marshal measurement message", func(t *testing.T) {
		msg := WSMessage{
			Type: EventTypeMeasurement,
			Data: MeasurementData{
				Turn:      1,
				Deff:      0.85,
				Beta:      0.72,
				Alignment: 0.91,
				Cpair:     0.88,
				Sender:    "agent-a",
				Receiver:  "agent-b",
			},
			Timestamp: "2025-01-01T00:00:00Z",
		}

		data, err := json.Marshal(msg)
		if err != nil {
			t.Fatalf("Failed to marshal message: %v", err)
		}

		var decoded WSMessage
		if err := json.Unmarshal(data, &decoded); err != nil {
			t.Fatalf("Failed to unmarshal message: %v", err)
		}

		if decoded.Type != EventTypeMeasurement {
			t.Errorf("Expected type %s, got %s", EventTypeMeasurement, decoded.Type)
		}
		if decoded.Timestamp != "2025-01-01T00:00:00Z" {
			t.Errorf("Expected timestamp 2025-01-01T00:00:00Z, got %s", decoded.Timestamp)
		}
	})

	t.Run("marshal subscribe message", func(t *testing.T) {
		msg := WSMessage{
			Type:     EventTypeSubscribe,
			Channels: []string{ChannelMeasurements, ChannelMessages},
		}

		data, err := json.Marshal(msg)
		if err != nil {
			t.Fatalf("Failed to marshal message: %v", err)
		}

		var decoded WSMessage
		if err := json.Unmarshal(data, &decoded); err != nil {
			t.Fatalf("Failed to unmarshal message: %v", err)
		}

		if decoded.Type != EventTypeSubscribe {
			t.Errorf("Expected type %s, got %s", EventTypeSubscribe, decoded.Type)
		}
		if len(decoded.Channels) != 2 {
			t.Errorf("Expected 2 channels, got %d", len(decoded.Channels))
		}
	})
}

// -----------------------------------------------------------------------------
// Hub Broadcast Tests
// -----------------------------------------------------------------------------

func TestHub_Broadcast(t *testing.T) {
	hub := NewHub()
	go hub.Run()
	defer hub.Stop()

	t.Run("broadcast to empty hub", func(t *testing.T) {
		msg := &WSMessage{
			Type: EventTypeMeasurement,
			Data: MeasurementData{Turn: 1, Deff: 0.5},
		}

		err := hub.Broadcast(msg)
		if err != nil {
			t.Errorf("Expected no error broadcasting to empty hub, got %v", err)
		}
	})

	t.Run("broadcast measurement", func(t *testing.T) {
		data := &MeasurementData{
			Turn:      1,
			Deff:      0.85,
			Beta:      0.72,
			Alignment: 0.91,
			Cpair:     0.88,
		}

		err := hub.BroadcastMeasurement(data)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
	})

	t.Run("broadcast message", func(t *testing.T) {
		data := &MessageData{
			Agent:   "test-agent",
			Content: "Hello, world!",
			Turn:    1,
		}

		err := hub.BroadcastMessage(data)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
	})

	t.Run("broadcast backend status", func(t *testing.T) {
		data := &BackendStatusData{
			Name:      "test-backend",
			Available: true,
			Capabilities: map[string]interface{}{
				"contextLimit": 32768,
			},
		}

		err := hub.BroadcastBackendStatus(data)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
	})

	t.Run("broadcast model status", func(t *testing.T) {
		data := &ModelStatusData{
			Name:   "test-model",
			Loaded: true,
			Memory: 4096,
		}

		err := hub.BroadcastModelStatus(data)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
	})

	t.Run("broadcast resource update", func(t *testing.T) {
		data := &ResourceUpdateData{
			GPUMemory:  8192,
			GPUUtil:    75,
			QueueDepth: 3,
		}

		err := hub.BroadcastResourceUpdate(data)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
	})
}

// -----------------------------------------------------------------------------
// WebSocket Handler Tests
// -----------------------------------------------------------------------------

func TestNewWebSocketHandler(t *testing.T) {
	hub := NewHub()
	handler := NewWebSocketHandler(hub)

	if handler == nil {
		t.Fatal("Expected handler to be created")
	}
	if handler.hub != hub {
		t.Error("Expected handler.hub to be set")
	}
}

func TestWebSocketHandler_HandleFunc(t *testing.T) {
	hub := NewHub()
	handler := NewWebSocketHandler(hub)

	fn := handler.HandleFunc()
	if fn == nil {
		t.Error("Expected HandlerFunc to be returned")
	}
}

// -----------------------------------------------------------------------------
// Integration Tests with HTTP Test Server
// -----------------------------------------------------------------------------

func TestWebSocket_Integration(t *testing.T) {
	hub := NewHub()
	go hub.Run()
	defer hub.Stop()

	handler := NewWebSocketHandler(hub)

	// Create test server
	server := httptest.NewServer(handler)
	defer server.Close()

	// Convert http URL to ws URL
	wsURL := "ws" + strings.TrimPrefix(server.URL, "http")

	t.Run("client can connect", func(t *testing.T) {
		conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
		if err != nil {
			t.Fatalf("Failed to connect: %v", err)
		}
		defer conn.Close()

		// Give time for registration
		time.Sleep(50 * time.Millisecond)

		if hub.ClientCount() != 1 {
			t.Errorf("Expected 1 connected client, got %d", hub.ClientCount())
		}
	})

	t.Run("client can disconnect", func(t *testing.T) {
		conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
		if err != nil {
			t.Fatalf("Failed to connect: %v", err)
		}

		// Give time for registration
		time.Sleep(50 * time.Millisecond)

		initialCount := hub.ClientCount()

		// Close connection
		conn.Close()

		// Give time for unregistration
		time.Sleep(50 * time.Millisecond)

		if hub.ClientCount() >= initialCount {
			t.Errorf("Expected client count to decrease after disconnect")
		}
	})

	t.Run("client can send subscribe message", func(t *testing.T) {
		conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
		if err != nil {
			t.Fatalf("Failed to connect: %v", err)
		}
		defer conn.Close()

		// Send subscribe message
		msg := WSMessage{
			Type:     EventTypeSubscribe,
			Channels: []string{ChannelMeasurements, ChannelMessages},
		}

		if err := conn.WriteJSON(msg); err != nil {
			t.Fatalf("Failed to send message: %v", err)
		}

		// Give time for processing
		time.Sleep(50 * time.Millisecond)
	})

	t.Run("client can send ping and receive pong", func(t *testing.T) {
		conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
		if err != nil {
			t.Fatalf("Failed to connect: %v", err)
		}
		defer conn.Close()

		// Set read deadline
		conn.SetReadDeadline(time.Now().Add(2 * time.Second))

		// Send ping message
		pingMsg := WSMessage{
			Type:      EventTypePing,
			Timestamp: time.Now().UTC().Format(time.RFC3339),
		}

		if err := conn.WriteJSON(pingMsg); err != nil {
			t.Fatalf("Failed to send ping: %v", err)
		}

		// Read pong response
		var pongMsg WSMessage
		if err := conn.ReadJSON(&pongMsg); err != nil {
			t.Fatalf("Failed to read pong: %v", err)
		}

		if pongMsg.Type != EventTypePong {
			t.Errorf("Expected pong message, got %s", pongMsg.Type)
		}
		if pongMsg.Timestamp == "" {
			t.Error("Expected timestamp in pong message")
		}
	})

	t.Run("client receives error for invalid JSON", func(t *testing.T) {
		conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
		if err != nil {
			t.Fatalf("Failed to connect: %v", err)
		}
		defer conn.Close()

		// Set read deadline
		conn.SetReadDeadline(time.Now().Add(2 * time.Second))

		// Send invalid JSON
		if err := conn.WriteMessage(websocket.TextMessage, []byte("not valid json")); err != nil {
			t.Fatalf("Failed to send message: %v", err)
		}

		// Read error response
		var errMsg WSMessage
		if err := conn.ReadJSON(&errMsg); err != nil {
			t.Fatalf("Failed to read error: %v", err)
		}

		if errMsg.Type != EventTypeError {
			t.Errorf("Expected error message, got %s", errMsg.Type)
		}
	})

	t.Run("client receives error for empty subscribe channels", func(t *testing.T) {
		conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
		if err != nil {
			t.Fatalf("Failed to connect: %v", err)
		}
		defer conn.Close()

		// Set read deadline
		conn.SetReadDeadline(time.Now().Add(2 * time.Second))

		// Send subscribe with no channels
		msg := WSMessage{
			Type:     EventTypeSubscribe,
			Channels: []string{},
		}

		if err := conn.WriteJSON(msg); err != nil {
			t.Fatalf("Failed to send message: %v", err)
		}

		// Read error response
		var errMsg WSMessage
		if err := conn.ReadJSON(&errMsg); err != nil {
			t.Fatalf("Failed to read error: %v", err)
		}

		if errMsg.Type != EventTypeError {
			t.Errorf("Expected error message, got %s", errMsg.Type)
		}
	})
}

// -----------------------------------------------------------------------------
// Broadcast to Subscribers Test
// -----------------------------------------------------------------------------

func TestHub_BroadcastToChannel(t *testing.T) {
	hub := NewHub()
	go hub.Run()
	defer hub.Stop()

	handler := NewWebSocketHandler(hub)
	server := httptest.NewServer(handler)
	defer server.Close()

	wsURL := "ws" + strings.TrimPrefix(server.URL, "http")

	t.Run("only subscribers receive channel messages", func(t *testing.T) {
		// Connect client 1 - subscribed to measurements
		conn1, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
		if err != nil {
			t.Fatalf("Failed to connect client 1: %v", err)
		}
		defer conn1.Close()

		// Connect client 2 - subscribed to status
		conn2, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
		if err != nil {
			t.Fatalf("Failed to connect client 2: %v", err)
		}
		defer conn2.Close()

		// Wait for connections
		time.Sleep(50 * time.Millisecond)

		// Subscribe client 1 to measurements
		if err := conn1.WriteJSON(WSMessage{
			Type:     EventTypeSubscribe,
			Channels: []string{ChannelMeasurements},
		}); err != nil {
			t.Fatalf("Failed to subscribe client 1: %v", err)
		}

		// Subscribe client 2 to status
		if err := conn2.WriteJSON(WSMessage{
			Type:     EventTypeSubscribe,
			Channels: []string{ChannelStatus},
		}); err != nil {
			t.Fatalf("Failed to subscribe client 2: %v", err)
		}

		// Wait for subscriptions
		time.Sleep(50 * time.Millisecond)

		// Broadcast measurement
		hub.BroadcastMeasurement(&MeasurementData{
			Turn: 1,
			Deff: 0.85,
		})

		// Client 1 should receive the message
		conn1.SetReadDeadline(time.Now().Add(500 * time.Millisecond))
		var msg1 WSMessage
		err = conn1.ReadJSON(&msg1)
		if err != nil {
			t.Errorf("Client 1 should have received measurement, got error: %v", err)
		} else if msg1.Type != EventTypeMeasurement {
			t.Errorf("Expected measurement message, got %s", msg1.Type)
		}

		// Client 2 should NOT receive the message (timeout expected)
		conn2.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
		var msg2 WSMessage
		err = conn2.ReadJSON(&msg2)
		if err == nil {
			t.Errorf("Client 2 should NOT have received measurement, but got: %s", msg2.Type)
		}
	})
}

// -----------------------------------------------------------------------------
// Concurrency Tests
// -----------------------------------------------------------------------------

func TestHub_Concurrent(t *testing.T) {
	hub := NewHub()
	go hub.Run()
	defer hub.Stop()

	handler := NewWebSocketHandler(hub)
	server := httptest.NewServer(handler)
	defer server.Close()

	wsURL := "ws" + strings.TrimPrefix(server.URL, "http")

	const numClients = 20
	var wg sync.WaitGroup

	// Connect multiple clients concurrently
	connections := make([]*websocket.Conn, numClients)
	errors := make([]error, numClients)

	for i := 0; i < numClients; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
			connections[idx] = conn
			errors[idx] = err
		}(i)
	}

	wg.Wait()

	// Check that all connections succeeded
	successCount := 0
	for i, err := range errors {
		if err != nil {
			t.Errorf("Client %d failed to connect: %v", i, err)
		} else {
			successCount++
		}
	}

	// Wait for all registrations
	time.Sleep(100 * time.Millisecond)

	if hub.ClientCount() != successCount {
		t.Errorf("Expected %d connected clients, got %d", successCount, hub.ClientCount())
	}

	// Cleanup
	for _, conn := range connections {
		if conn != nil {
			conn.Close()
		}
	}
}

func TestHub_ConcurrentBroadcast(t *testing.T) {
	hub := NewHub()
	go hub.Run()
	defer hub.Stop()

	// Broadcast many messages concurrently
	const numMessages = 100
	var wg sync.WaitGroup

	for i := 0; i < numMessages; i++ {
		wg.Add(1)
		go func(turn int) {
			defer wg.Done()
			hub.BroadcastMeasurement(&MeasurementData{
				Turn: turn,
				Deff: float64(turn) / 100,
			})
		}(i)
	}

	wg.Wait()
	// If we get here without deadlock or panic, the test passed
}

// -----------------------------------------------------------------------------
// Data Type Tests
// -----------------------------------------------------------------------------

func TestMeasurementData_JSON(t *testing.T) {
	data := MeasurementData{
		Turn:      5,
		Deff:      0.85,
		Beta:      0.72,
		Alignment: 0.91,
		Cpair:     0.88,
		Sender:    "agent-a",
		Receiver:  "agent-b",
	}

	bytes, err := json.Marshal(data)
	if err != nil {
		t.Fatalf("Failed to marshal: %v", err)
	}

	var decoded MeasurementData
	if err := json.Unmarshal(bytes, &decoded); err != nil {
		t.Fatalf("Failed to unmarshal: %v", err)
	}

	if decoded.Turn != 5 {
		t.Errorf("Expected turn 5, got %d", decoded.Turn)
	}
	if decoded.Deff != 0.85 {
		t.Errorf("Expected deff 0.85, got %f", decoded.Deff)
	}
	if decoded.Sender != "agent-a" {
		t.Errorf("Expected sender 'agent-a', got %s", decoded.Sender)
	}
}

func TestMessageData_JSON(t *testing.T) {
	data := MessageData{
		Agent:   "test-agent",
		Content: "Hello, world!",
		Turn:    3,
	}

	bytes, err := json.Marshal(data)
	if err != nil {
		t.Fatalf("Failed to marshal: %v", err)
	}

	var decoded MessageData
	if err := json.Unmarshal(bytes, &decoded); err != nil {
		t.Fatalf("Failed to unmarshal: %v", err)
	}

	if decoded.Agent != "test-agent" {
		t.Errorf("Expected agent 'test-agent', got %s", decoded.Agent)
	}
	if decoded.Content != "Hello, world!" {
		t.Errorf("Expected content 'Hello, world!', got %s", decoded.Content)
	}
	if decoded.Turn != 3 {
		t.Errorf("Expected turn 3, got %d", decoded.Turn)
	}
}

func TestBackendStatusData_JSON(t *testing.T) {
	data := BackendStatusData{
		Name:      "loom-backend",
		Available: true,
		Capabilities: map[string]interface{}{
			"contextLimit":      32768,
			"supportsTools":     true,
			"supportsStreaming": true,
		},
	}

	bytes, err := json.Marshal(data)
	if err != nil {
		t.Fatalf("Failed to marshal: %v", err)
	}

	var decoded BackendStatusData
	if err := json.Unmarshal(bytes, &decoded); err != nil {
		t.Fatalf("Failed to unmarshal: %v", err)
	}

	if decoded.Name != "loom-backend" {
		t.Errorf("Expected name 'loom-backend', got %s", decoded.Name)
	}
	if !decoded.Available {
		t.Error("Expected available to be true")
	}
	if decoded.Capabilities == nil {
		t.Error("Expected capabilities to be set")
	}
}

func TestModelStatusData_JSON(t *testing.T) {
	data := ModelStatusData{
		Name:   "llama-3.1",
		Loaded: true,
		Memory: 16384,
	}

	bytes, err := json.Marshal(data)
	if err != nil {
		t.Fatalf("Failed to marshal: %v", err)
	}

	var decoded ModelStatusData
	if err := json.Unmarshal(bytes, &decoded); err != nil {
		t.Fatalf("Failed to unmarshal: %v", err)
	}

	if decoded.Name != "llama-3.1" {
		t.Errorf("Expected name 'llama-3.1', got %s", decoded.Name)
	}
	if !decoded.Loaded {
		t.Error("Expected loaded to be true")
	}
	if decoded.Memory != 16384 {
		t.Errorf("Expected memory 16384, got %d", decoded.Memory)
	}
}

func TestResourceUpdateData_JSON(t *testing.T) {
	data := ResourceUpdateData{
		GPUMemory:  8192,
		GPUUtil:    75.5,
		QueueDepth: 3,
	}

	bytes, err := json.Marshal(data)
	if err != nil {
		t.Fatalf("Failed to marshal: %v", err)
	}

	var decoded ResourceUpdateData
	if err := json.Unmarshal(bytes, &decoded); err != nil {
		t.Fatalf("Failed to unmarshal: %v", err)
	}

	if decoded.GPUMemory != 8192 {
		t.Errorf("Expected GPUMemory 8192, got %d", decoded.GPUMemory)
	}
	if decoded.GPUUtil != 75.5 {
		t.Errorf("Expected GPUUtil 75.5, got %f", decoded.GPUUtil)
	}
	if decoded.QueueDepth != 3 {
		t.Errorf("Expected QueueDepth 3, got %d", decoded.QueueDepth)
	}
}

// -----------------------------------------------------------------------------
// Constants Tests
// -----------------------------------------------------------------------------

func TestWebSocketConstants(t *testing.T) {
	// Verify channel names
	if ChannelMeasurements != "measurements" {
		t.Errorf("Expected ChannelMeasurements to be 'measurements', got %s", ChannelMeasurements)
	}
	if ChannelMessages != "messages" {
		t.Errorf("Expected ChannelMessages to be 'messages', got %s", ChannelMessages)
	}
	if ChannelStatus != "status" {
		t.Errorf("Expected ChannelStatus to be 'status', got %s", ChannelStatus)
	}
	if ChannelResources != "resources" {
		t.Errorf("Expected ChannelResources to be 'resources', got %s", ChannelResources)
	}

	// Verify event types
	if EventTypeMeasurement != "measurement" {
		t.Errorf("Expected EventTypeMeasurement to be 'measurement', got %s", EventTypeMeasurement)
	}
	if EventTypeMessage != "message" {
		t.Errorf("Expected EventTypeMessage to be 'message', got %s", EventTypeMessage)
	}
	if EventTypeBackendStatus != "backend_status" {
		t.Errorf("Expected EventTypeBackendStatus to be 'backend_status', got %s", EventTypeBackendStatus)
	}
	if EventTypeModelStatus != "model_status" {
		t.Errorf("Expected EventTypeModelStatus to be 'model_status', got %s", EventTypeModelStatus)
	}
	if EventTypeResourceUpdate != "resource_update" {
		t.Errorf("Expected EventTypeResourceUpdate to be 'resource_update', got %s", EventTypeResourceUpdate)
	}
	if EventTypePong != "pong" {
		t.Errorf("Expected EventTypePong to be 'pong', got %s", EventTypePong)
	}
	if EventTypePing != "ping" {
		t.Errorf("Expected EventTypePing to be 'ping', got %s", EventTypePing)
	}
	if EventTypeSubscribe != "subscribe" {
		t.Errorf("Expected EventTypeSubscribe to be 'subscribe', got %s", EventTypeSubscribe)
	}
	if EventTypeError != "error" {
		t.Errorf("Expected EventTypeError to be 'error', got %s", EventTypeError)
	}
}

// -----------------------------------------------------------------------------
// Upgrader Tests
// -----------------------------------------------------------------------------

func TestSetUpgraderCheckOrigin(t *testing.T) {
	// Custom origin check that rejects everything
	SetUpgraderCheckOrigin(func(r *http.Request) bool {
		return false
	})

	hub := NewHub()
	go hub.Run()
	defer hub.Stop()

	handler := NewWebSocketHandler(hub)
	server := httptest.NewServer(handler)
	defer server.Close()

	wsURL := "ws" + strings.TrimPrefix(server.URL, "http")

	// Connection should fail due to origin check
	_, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err == nil {
		t.Error("Expected connection to fail with custom origin check")
	}

	// Reset to allow all origins
	SetUpgraderCheckOrigin(func(r *http.Request) bool {
		return true
	})
}

// -----------------------------------------------------------------------------
// Edge Cases
// -----------------------------------------------------------------------------

func TestClient_HandleMessage_UnknownType(t *testing.T) {
	hub := NewHub()
	go hub.Run()
	defer hub.Stop()

	handler := NewWebSocketHandler(hub)
	server := httptest.NewServer(handler)
	defer server.Close()

	wsURL := "ws" + strings.TrimPrefix(server.URL, "http")

	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("Failed to connect: %v", err)
	}
	defer conn.Close()

	// Send unknown message type
	msg := WSMessage{
		Type: "unknown_type",
		Data: map[string]string{"foo": "bar"},
	}

	if err := conn.WriteJSON(msg); err != nil {
		t.Fatalf("Failed to send message: %v", err)
	}

	// Should not receive an error (unknown types are logged but not responded to)
	// Give time for processing
	time.Sleep(50 * time.Millisecond)

	// If we get here without panic or hang, the test passed
}

func TestClient_SubscribeToInvalidChannel(t *testing.T) {
	hub := NewHub()
	go hub.Run()
	defer hub.Stop()

	handler := NewWebSocketHandler(hub)
	server := httptest.NewServer(handler)
	defer server.Close()

	wsURL := "ws" + strings.TrimPrefix(server.URL, "http")

	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("Failed to connect: %v", err)
	}
	defer conn.Close()

	// Subscribe with one valid and one invalid channel
	msg := WSMessage{
		Type:     EventTypeSubscribe,
		Channels: []string{ChannelMeasurements, "invalid_channel"},
	}

	if err := conn.WriteJSON(msg); err != nil {
		t.Fatalf("Failed to send message: %v", err)
	}

	// Give time for processing
	time.Sleep(50 * time.Millisecond)

	// The valid channel should still work
	hub.BroadcastMeasurement(&MeasurementData{Turn: 1, Deff: 0.5})

	conn.SetReadDeadline(time.Now().Add(500 * time.Millisecond))
	var received WSMessage
	err = conn.ReadJSON(&received)
	if err != nil {
		t.Errorf("Should have received measurement on valid channel: %v", err)
	}
}
