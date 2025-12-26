# Consolidate duplicated HiddenState type definitions

## Overview

The HiddenState struct is defined in two packages with identical fields: yarn.HiddenState and backend.HiddenState. A convertHiddenState function exists in concepts/extractor.go to convert between them. This duplication creates maintenance burden and risks the types drifting apart.

## Rationale

Type duplication violates DRY (Don't Repeat Yourself) and creates maintenance overhead. When one definition is updated, the other must be manually updated too. A shared type definition ensures consistency and eliminates conversion code.

---
*This spec was created from ideation and is pending detailed specification.*
