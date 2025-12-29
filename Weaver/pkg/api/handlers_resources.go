// Package api provides the HTTP/WebSocket server for the Weaver web UI.
package api

import (
	"bufio"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// ResourcesHandler handles system resource API requests.
type ResourcesHandler struct{}

// NewResourcesHandler creates a new ResourcesHandler.
func NewResourcesHandler() *ResourcesHandler {
	return &ResourcesHandler{}
}

// RegisterRoutes registers the resource API routes on the router.
func (h *ResourcesHandler) RegisterRoutes(router *Router) {
	router.GET("/api/resources/gpus", h.ListGPUs)
}

// -----------------------------------------------------------------------------
// API Response Types
// -----------------------------------------------------------------------------

// GPUListResponse is the JSON response for GET /api/resources/gpus.
type GPUListResponse struct {
	GPUs  []GPUInfo `json:"gpus"`
	Total int       `json:"total"`
}

// GPUInfo represents information about a single GPU.
type GPUInfo struct {
	Index       int    `json:"index"`       // GPU device index (0, 1, 2, ...)
	Name        string `json:"name"`        // GPU model name
	MemoryTotal int64  `json:"memoryTotal"` // Total memory in MB
	MemoryFree  int64  `json:"memoryFree"`  // Free memory in MB
	MemoryUsed  int64  `json:"memoryUsed"`  // Used memory in MB
	Utilization int    `json:"utilization"` // GPU utilization percentage (0-100)
	Available   bool   `json:"available"`   // Whether GPU appears available for use
}

// -----------------------------------------------------------------------------
// Handlers
// -----------------------------------------------------------------------------

// ListGPUs handles GET /api/resources/gpus.
// It returns a list of available NVIDIA GPUs on the system.
func (h *ResourcesHandler) ListGPUs(w http.ResponseWriter, r *http.Request) {
	gpus, err := detectNVIDIAGPUs()
	if err != nil {
		// Return empty list if no GPUs found or nvidia-smi not available
		response := GPUListResponse{
			GPUs:  []GPUInfo{},
			Total: 0,
		}
		WriteJSON(w, http.StatusOK, response)
		return
	}

	response := GPUListResponse{
		GPUs:  gpus,
		Total: len(gpus),
	}

	WriteJSON(w, http.StatusOK, response)
}

// -----------------------------------------------------------------------------
// GPU Detection
// -----------------------------------------------------------------------------

// detectNVIDIAGPUs uses nvidia-smi to detect available NVIDIA GPUs.
func detectNVIDIAGPUs() ([]GPUInfo, error) {
	// Try nvidia-smi first
	gpus, err := detectWithNvidiaSmi()
	if err == nil && len(gpus) > 0 {
		return gpus, nil
	}

	// Fallback to sysfs detection on Linux
	gpus, err = detectWithSysfs()
	if err == nil && len(gpus) > 0 {
		return gpus, nil
	}

	return nil, err
}

// detectWithNvidiaSmi uses nvidia-smi command to get GPU info.
func detectWithNvidiaSmi() ([]GPUInfo, error) {
	// Check if nvidia-smi exists
	nvidiaSmiPath, err := exec.LookPath("nvidia-smi")
	if err != nil {
		return nil, err
	}

	// Query GPU info in CSV format
	// --query-gpu: index, name, memory.total, memory.free, memory.used, utilization.gpu
	cmd := exec.Command(nvidiaSmiPath,
		"--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu",
		"--format=csv,noheader,nounits")

	output, err := cmd.Output()
	if err != nil {
		return nil, err
	}

	return parseNvidiaSmiOutput(string(output))
}

// parseNvidiaSmiOutput parses the CSV output from nvidia-smi.
func parseNvidiaSmiOutput(output string) ([]GPUInfo, error) {
	var gpus []GPUInfo

	scanner := bufio.NewScanner(strings.NewReader(output))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		// Parse CSV: index, name, memory.total, memory.free, memory.used, utilization.gpu
		parts := strings.Split(line, ", ")
		if len(parts) < 6 {
			continue
		}

		index, _ := strconv.Atoi(strings.TrimSpace(parts[0]))
		name := strings.TrimSpace(parts[1])
		memTotal, _ := strconv.ParseInt(strings.TrimSpace(parts[2]), 10, 64)
		memFree, _ := strconv.ParseInt(strings.TrimSpace(parts[3]), 10, 64)
		memUsed, _ := strconv.ParseInt(strings.TrimSpace(parts[4]), 10, 64)
		utilization, _ := strconv.Atoi(strings.TrimSpace(parts[5]))

		// Consider GPU available if utilization < 80% and has free memory
		available := utilization < 80 && memFree > 1000 // At least 1GB free

		gpus = append(gpus, GPUInfo{
			Index:       index,
			Name:        name,
			MemoryTotal: memTotal,
			MemoryFree:  memFree,
			MemoryUsed:  memUsed,
			Utilization: utilization,
			Available:   available,
		})
	}

	return gpus, scanner.Err()
}

// detectWithSysfs uses Linux sysfs to detect NVIDIA GPUs.
// This is a fallback when nvidia-smi is not available.
func detectWithSysfs() ([]GPUInfo, error) {
	// Look for NVIDIA devices in /sys/class/drm
	drmPath := "/sys/class/drm"
	entries, err := os.ReadDir(drmPath)
	if err != nil {
		return nil, err
	}

	var gpus []GPUInfo
	gpuIndex := 0

	// Pattern for card directories (card0, card1, etc.)
	cardPattern := regexp.MustCompile(`^card(\d+)$`)

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		matches := cardPattern.FindStringSubmatch(entry.Name())
		if matches == nil {
			continue
		}

		// Check if this is an NVIDIA device by reading vendor
		vendorPath := filepath.Join(drmPath, entry.Name(), "device", "vendor")
		vendorData, err := os.ReadFile(vendorPath)
		if err != nil {
			continue
		}

		vendor := strings.TrimSpace(string(vendorData))
		// NVIDIA vendor ID is 0x10de
		if vendor != "0x10de" {
			continue
		}

		// Try to get device name
		name := "NVIDIA GPU"
		devicePath := filepath.Join(drmPath, entry.Name(), "device", "device")
		if deviceData, err := os.ReadFile(devicePath); err == nil {
			deviceID := strings.TrimSpace(string(deviceData))
			name = "NVIDIA GPU (Device " + deviceID + ")"
		}

		gpus = append(gpus, GPUInfo{
			Index:       gpuIndex,
			Name:        name,
			MemoryTotal: 0, // Unknown without nvidia-smi
			MemoryFree:  0,
			MemoryUsed:  0,
			Utilization: 0,
			Available:   true, // Assume available
		})

		gpuIndex++
	}

	return gpus, nil
}
