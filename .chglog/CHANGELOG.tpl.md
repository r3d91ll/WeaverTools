# Changelog

All notable changes to WeaverTools will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

{{ if .Versions -}}
{{ range .Versions }}
<a name="{{ .Tag.Name }}"></a>
## {{ if .Tag.Previous }}[{{ .Tag.Name }}]({{ $.Info.RepositoryURL }}/compare/{{ .Tag.Previous.Name }}...{{ .Tag.Name }}){{ else }}{{ .Tag.Name }}{{ end }} - {{ datetime "2006-01-02" .Tag.Date }}

{{ if .CommitGroups -}}
{{ range .CommitGroups -}}
### {{ .Title }}

{{ range .Commits -}}
{{ if .Scope -}}
- **{{ .Scope }}**: {{ .Subject }}
{{ else -}}
- {{ .Subject }}
{{ end -}}
{{ end }}
{{ end -}}
{{ end -}}

{{ if .NoteGroups -}}
{{ range .NoteGroups -}}
### {{ .Title }}

{{ range .Notes -}}
{{ .Body }}
{{ end -}}
{{ end -}}
{{ end -}}
{{ end -}}
{{ end -}}

---

## Versioning Policy

WeaverTools follows [Semantic Versioning 2.0.0](https://semver.org/):

- **MAJOR** version for incompatible API changes
- **MINOR** version for backward-compatible functionality additions
- **PATCH** version for backward-compatible bug fixes

### Deprecation Policy

All deprecated APIs receive a minimum **6-month notice** before removal:

1. Deprecated features are marked with `// Deprecated:` godoc comments
2. Deprecated features appear in the CHANGELOG under "Deprecated"
3. Removal is announced at least one major version in advance
4. See [VERSIONING.md](docs/VERSIONING.md) for full deprecation policy
