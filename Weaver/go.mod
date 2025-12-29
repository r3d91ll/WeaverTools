module github.com/r3d91ll/weaver

go 1.23.4

require (
	github.com/chzyer/readline v1.5.1
	github.com/google/uuid v1.6.0
	github.com/gorilla/websocket v1.5.3
	github.com/r3d91ll/wool v0.0.0
	github.com/r3d91ll/yarn v0.0.0
	golang.org/x/term v0.27.0
	gopkg.in/yaml.v3 v3.0.1
)

require golang.org/x/sys v0.28.0 // indirect

replace (
	github.com/r3d91ll/wool => ../Wool
	github.com/r3d91ll/yarn => ../Yarn
)
