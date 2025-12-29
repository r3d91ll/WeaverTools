// Package api provides the HTTP/WebSocket server for the Weaver web UI.
package api

import (
	"encoding/json"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

// -----------------------------------------------------------------------------
// WebSocket Constants
// -----------------------------------------------------------------------------

const (
	// Time allowed to write a message to the peer.
	writeWait = 10 * time.Second

	// Time allowed to read the next pong message from the peer.
	pongWait = 60 * time.Second

	// Send pings to peer with this period. Must be less than pongWait.
	pingPeriod = (pongWait * 9) / 10

	// Maximum message size allowed from peer.
	maxMessageSize = 8192

	// Size of client send buffer.
	sendBufferSize = 256
)

// Channel names for subscriptions
const (
	ChannelMeasurements = "measurements"
	ChannelMessages     = "messages"
	ChannelStatus       = "status"
	ChannelResources    = "resources"
)

// Event types for WebSocket messages
const (
	EventTypeMeasurement    = "measurement"
	EventTypeMessage        = "message"
	EventTypeBackendStatus  = "backend_status"
	EventTypeModelStatus    = "model_status"
	EventTypeResourceUpdate = "resource_update"
	EventTypePong           = "pong"
	EventTypeSubscribe      = "subscribe"
	EventTypePing           = "ping"
	EventTypeError          = "error"
)

// -----------------------------------------------------------------------------
// WebSocket Message Types
// -----------------------------------------------------------------------------

// WSMessage is the standard WebSocket message envelope.
type WSMessage struct {
	Type      string      `json:"type"`
	Data      interface{} `json:"data,omitempty"`
	Timestamp string      `json:"timestamp,omitempty"`
	Channels  []string    `json:"channels,omitempty"` // For subscribe messages
}

// MeasurementData represents measurement event data.
type MeasurementData struct {
	Turn      int     `json:"turn"`
	Deff      float64 `json:"deff"`
	Beta      float64 `json:"beta"`
	Alignment float64 `json:"alignment"`
	Cpair     float64 `json:"cpair"`
	Sender    string  `json:"sender,omitempty"`
	Receiver  string  `json:"receiver,omitempty"`
}

// MessageData represents a chat message event.
type MessageData struct {
	Agent   string `json:"agent"`
	Content string `json:"content"`
	Turn    int    `json:"turn"`
}

// BackendStatusData represents backend status event data.
type BackendStatusData struct {
	Name         string            `json:"name"`
	Available    bool              `json:"available"`
	Capabilities map[string]interface{} `json:"capabilities,omitempty"`
}

// ModelStatusData represents model status event data.
type ModelStatusData struct {
	Name   string `json:"name"`
	Loaded bool   `json:"loaded"`
	Memory int64  `json:"memory,omitempty"`
}

// ResourceUpdateData represents resource usage event data.
type ResourceUpdateData struct {
	GPUMemory  int64   `json:"gpuMemory"`
	GPUUtil    float64 `json:"gpuUtil"`
	QueueDepth int     `json:"queueDepth"`
}

// -----------------------------------------------------------------------------
// WebSocket Upgrader
// -----------------------------------------------------------------------------

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	// Allow all origins in development; configure in production
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}

// SetUpgraderCheckOrigin allows customizing the origin check function.
func SetUpgraderCheckOrigin(fn func(*http.Request) bool) {
	upgrader.CheckOrigin = fn
}

// -----------------------------------------------------------------------------
// Client
// -----------------------------------------------------------------------------

// Client represents a single WebSocket client connection.
type Client struct {
	hub  *Hub
	conn *websocket.Conn
	send chan []byte

	// subscriptions tracks which channels this client is subscribed to
	subscriptions map[string]bool
	subMu         sync.RWMutex
}

// NewClient creates a new WebSocket client.
func NewClient(hub *Hub, conn *websocket.Conn) *Client {
	return &Client{
		hub:           hub,
		conn:          conn,
		send:          make(chan []byte, sendBufferSize),
		subscriptions: make(map[string]bool),
	}
}

// Subscribe adds a channel subscription for this client.
func (c *Client) Subscribe(channels ...string) {
	c.subMu.Lock()
	defer c.subMu.Unlock()
	for _, ch := range channels {
		c.subscriptions[ch] = true
	}
}

// Unsubscribe removes a channel subscription for this client.
func (c *Client) Unsubscribe(channels ...string) {
	c.subMu.Lock()
	defer c.subMu.Unlock()
	for _, ch := range channels {
		delete(c.subscriptions, ch)
	}
}

// IsSubscribed checks if the client is subscribed to a channel.
func (c *Client) IsSubscribed(channel string) bool {
	c.subMu.RLock()
	defer c.subMu.RUnlock()
	return c.subscriptions[channel]
}

// Subscriptions returns a copy of the client's subscribed channels.
func (c *Client) Subscriptions() []string {
	c.subMu.RLock()
	defer c.subMu.RUnlock()
	channels := make([]string, 0, len(c.subscriptions))
	for ch := range c.subscriptions {
		channels = append(channels, ch)
	}
	return channels
}

// readPump pumps messages from the WebSocket connection to the hub.
// The application runs readPump in a per-connection goroutine.
func (c *Client) readPump() {
	defer func() {
		c.hub.unregister <- c
		c.conn.Close()
	}()

	c.conn.SetReadLimit(maxMessageSize)
	c.conn.SetReadDeadline(time.Now().Add(pongWait))
	c.conn.SetPongHandler(func(string) error {
		c.conn.SetReadDeadline(time.Now().Add(pongWait))
		return nil
	})

	for {
		_, message, err := c.conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("[ws] read error: %v", err)
			}
			break
		}

		// Parse and handle client message
		c.handleMessage(message)
	}
}

// handleMessage processes an incoming message from the client.
func (c *Client) handleMessage(message []byte) {
	var msg WSMessage
	if err := json.Unmarshal(message, &msg); err != nil {
		c.sendError("invalid_json", "Failed to parse message")
		return
	}

	switch msg.Type {
	case EventTypeSubscribe:
		c.handleSubscribe(msg)
	case EventTypePing:
		c.handlePing(msg)
	default:
		// Unknown message type - log but don't error
		log.Printf("[ws] unknown message type: %s", msg.Type)
	}
}

// handleSubscribe processes a subscribe message.
func (c *Client) handleSubscribe(msg WSMessage) {
	if len(msg.Channels) == 0 {
		c.sendError("invalid_subscribe", "No channels specified")
		return
	}

	validChannels := make([]string, 0, len(msg.Channels))
	for _, ch := range msg.Channels {
		switch ch {
		case ChannelMeasurements, ChannelMessages, ChannelStatus, ChannelResources:
			validChannels = append(validChannels, ch)
		default:
			log.Printf("[ws] unknown channel: %s", ch)
		}
	}

	if len(validChannels) > 0 {
		c.Subscribe(validChannels...)
		log.Printf("[ws] client subscribed to: %v", validChannels)
	}
}

// handlePing processes a ping message and sends a pong response.
func (c *Client) handlePing(msg WSMessage) {
	pong := WSMessage{
		Type:      EventTypePong,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	}
	data, err := json.Marshal(pong)
	if err != nil {
		return
	}

	select {
	case c.send <- data:
	default:
		// Buffer full, drop the message
	}
}

// sendError sends an error message to the client.
func (c *Client) sendError(code, message string) {
	errMsg := WSMessage{
		Type: EventTypeError,
		Data: map[string]string{
			"code":    code,
			"message": message,
		},
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	}
	data, err := json.Marshal(errMsg)
	if err != nil {
		return
	}

	select {
	case c.send <- data:
	default:
		// Buffer full, drop the message
	}
}

// writePump pumps messages from the hub to the WebSocket connection.
// A goroutine running writePump is started for each connection.
func (c *Client) writePump() {
	ticker := time.NewTicker(pingPeriod)
	defer func() {
		ticker.Stop()
		c.conn.Close()
	}()

	for {
		select {
		case message, ok := <-c.send:
			c.conn.SetWriteDeadline(time.Now().Add(writeWait))
			if !ok {
				// The hub closed the channel.
				c.conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}

			w, err := c.conn.NextWriter(websocket.TextMessage)
			if err != nil {
				return
			}
			w.Write(message)

			// Add queued messages to the current WebSocket message.
			n := len(c.send)
			for i := 0; i < n; i++ {
				w.Write([]byte{'\n'})
				w.Write(<-c.send)
			}

			if err := w.Close(); err != nil {
				return
			}
		case <-ticker.C:
			c.conn.SetWriteDeadline(time.Now().Add(writeWait))
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

// -----------------------------------------------------------------------------
// Hub
// -----------------------------------------------------------------------------

// Hub maintains the set of active clients and broadcasts messages to them.
type Hub struct {
	// clients is the set of registered clients
	clients map[*Client]bool

	// broadcast is the channel for messages to broadcast to all clients
	broadcast chan []byte

	// register is the channel for new clients
	register chan *Client

	// unregister is the channel for disconnecting clients
	unregister chan *Client

	// mu protects the clients map
	mu sync.RWMutex

	// done signals the hub to stop
	done chan struct{}
}

// NewHub creates a new WebSocket hub.
func NewHub() *Hub {
	return &Hub{
		clients:    make(map[*Client]bool),
		broadcast:  make(chan []byte, 256),
		register:   make(chan *Client),
		unregister: make(chan *Client),
		done:       make(chan struct{}),
	}
}

// Run starts the hub's main loop.
func (h *Hub) Run() {
	for {
		select {
		case <-h.done:
			// Cleanup on shutdown
			h.mu.Lock()
			for client := range h.clients {
				close(client.send)
				delete(h.clients, client)
			}
			h.mu.Unlock()
			return

		case client := <-h.register:
			h.mu.Lock()
			h.clients[client] = true
			h.mu.Unlock()
			log.Printf("[ws] client connected (total: %d)", h.ClientCount())

		case client := <-h.unregister:
			h.mu.Lock()
			if _, ok := h.clients[client]; ok {
				delete(h.clients, client)
				close(client.send)
			}
			h.mu.Unlock()
			log.Printf("[ws] client disconnected (total: %d)", h.ClientCount())

		case message := <-h.broadcast:
			h.mu.RLock()
			for client := range h.clients {
				select {
				case client.send <- message:
				default:
					// Client buffer is full, close connection
					close(client.send)
					delete(h.clients, client)
				}
			}
			h.mu.RUnlock()
		}
	}
}

// Stop gracefully stops the hub.
func (h *Hub) Stop() {
	close(h.done)
}

// ClientCount returns the number of connected clients.
func (h *Hub) ClientCount() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.clients)
}

// Broadcast sends a message to all connected clients.
func (h *Hub) Broadcast(msg *WSMessage) error {
	data, err := json.Marshal(msg)
	if err != nil {
		return err
	}

	select {
	case h.broadcast <- data:
		return nil
	default:
		return nil // Buffer full, drop message silently
	}
}

// BroadcastToChannel sends a message to clients subscribed to a specific channel.
func (h *Hub) BroadcastToChannel(channel string, msg *WSMessage) error {
	data, err := json.Marshal(msg)
	if err != nil {
		return err
	}

	h.mu.RLock()
	defer h.mu.RUnlock()

	for client := range h.clients {
		if client.IsSubscribed(channel) {
			select {
			case client.send <- data:
			default:
				// Client buffer is full, skip
			}
		}
	}
	return nil
}

// BroadcastMeasurement sends a measurement event to subscribed clients.
func (h *Hub) BroadcastMeasurement(data *MeasurementData) error {
	msg := &WSMessage{
		Type:      EventTypeMeasurement,
		Data:      data,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	}
	return h.BroadcastToChannel(ChannelMeasurements, msg)
}

// BroadcastMessage sends a chat message event to subscribed clients.
func (h *Hub) BroadcastMessage(data *MessageData) error {
	msg := &WSMessage{
		Type:      EventTypeMessage,
		Data:      data,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	}
	return h.BroadcastToChannel(ChannelMessages, msg)
}

// BroadcastBackendStatus sends a backend status event to subscribed clients.
func (h *Hub) BroadcastBackendStatus(data *BackendStatusData) error {
	msg := &WSMessage{
		Type:      EventTypeBackendStatus,
		Data:      data,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	}
	return h.BroadcastToChannel(ChannelStatus, msg)
}

// BroadcastModelStatus sends a model status event to subscribed clients.
func (h *Hub) BroadcastModelStatus(data *ModelStatusData) error {
	msg := &WSMessage{
		Type:      EventTypeModelStatus,
		Data:      data,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	}
	return h.BroadcastToChannel(ChannelStatus, msg)
}

// BroadcastResourceUpdate sends a resource update event to subscribed clients.
func (h *Hub) BroadcastResourceUpdate(data *ResourceUpdateData) error {
	msg := &WSMessage{
		Type:      EventTypeResourceUpdate,
		Data:      data,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	}
	return h.BroadcastToChannel(ChannelResources, msg)
}

// -----------------------------------------------------------------------------
// HTTP Handler
// -----------------------------------------------------------------------------

// WebSocketHandler handles WebSocket upgrade requests.
type WebSocketHandler struct {
	hub *Hub
}

// NewWebSocketHandler creates a new WebSocket handler with the given hub.
func NewWebSocketHandler(hub *Hub) *WebSocketHandler {
	return &WebSocketHandler{hub: hub}
}

// ServeHTTP implements http.Handler for WebSocket connections.
func (h *WebSocketHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("[ws] upgrade error: %v", err)
		return
	}

	client := NewClient(h.hub, conn)
	h.hub.register <- client

	// Start the client's read and write pumps
	go client.writePump()
	go client.readPump()
}

// HandleFunc returns an http.HandlerFunc for WebSocket connections.
// This is a convenience method for use with the custom Router.
func (h *WebSocketHandler) HandleFunc() HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		h.ServeHTTP(w, r)
	}
}
