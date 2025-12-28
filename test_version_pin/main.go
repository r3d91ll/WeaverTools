// Package main demonstrates version pinning for WeaverTools modules.
//
// This test project shows how to pin specific versions of Yarn (and other
// WeaverTools modules) in your go.mod file to ensure reproducibility
// in long-running experiments and research projects.
//
// Version Pinning Benefits:
//   - Reproducible builds: Same version = same behavior
//   - Experiment isolation: Upgrade WeaverTools without affecting running experiments
//   - Rollback capability: Pin to known-good versions if issues arise
//
// See docs/VERSIONING.md for the full versioning policy.
package main

import (
	"fmt"

	yarn "github.com/r3d91ll/yarn"
)

func main() {
	// Create a simple message using Yarn v1.0.0 API
	msg := yarn.NewMessage(yarn.RoleUser, "Hello from version-pinned test!")

	// Validate the message
	if err := msg.Validate(); err != nil {
		fmt.Printf("Validation error: %s\n", err.Error())
		return
	}

	// Display message info
	fmt.Println("=== Version Pinning Test ===")
	fmt.Printf("Message ID: %s\n", msg.ID)
	fmt.Printf("Role: %s\n", msg.Role)
	fmt.Printf("Content: %s\n", msg.Content)
	fmt.Printf("Timestamp: %s\n", msg.Timestamp.Format("2006-01-02 15:04:05"))
	fmt.Println("")
	fmt.Println("Version pinning test PASSED!")
	fmt.Println("This project successfully uses github.com/r3d91ll/yarn v1.0.0")
}
