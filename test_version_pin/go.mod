module example/test_version_pin

go 1.23.4

// Version Pinning Example
// This demonstrates how to pin specific versions of WeaverTools modules
// in your experiment configurations for reproducibility.
//
// In production (when using published modules from GitHub):
//
//   require (
//       github.com/r3d91ll/yarn v1.0.0
//       github.com/r3d91ll/wool v1.0.0
//       github.com/r3d91ll/weaver v1.0.0
//   )
//
// For local development in the monorepo, we use replace directives
// to point to local module paths. These would be removed when
// consuming the published packages.

require (
	github.com/r3d91ll/yarn v1.0.0
)

// Required dependencies for yarn
require github.com/google/uuid v1.6.0 // indirect

// Development-only: Replace directives for local monorepo development
// Remove these when using published packages from GitHub
replace github.com/r3d91ll/yarn => ../Yarn
