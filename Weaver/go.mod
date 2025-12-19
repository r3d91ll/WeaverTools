module github.com/r3d91ll/weaver

go 1.23.4

require (
	github.com/chzyer/readline v1.5.1
	github.com/r3d91ll/wool v0.0.0
	github.com/r3d91ll/yarn v0.0.0
	gopkg.in/yaml.v3 v3.0.1
)

require (
	github.com/google/uuid v1.6.0 // indirect
	golang.org/x/sys v0.0.0-20220310020820-b874c991c1a5 // indirect
)

replace (
	github.com/r3d91ll/wool => ../Wool
	github.com/r3d91ll/yarn => ../Yarn
)
