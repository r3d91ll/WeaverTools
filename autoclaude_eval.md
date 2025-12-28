# Auto Claude Evaluation Notes

## Git Workflow Observations

- **Commit behavior**: When clicking "commit" on a Kanban task, Auto Claude only commits locally. The branch must be manually pushed upstream, and PR creation/merge handled outside the application.

- **Branch staleness issue**: Auto Claude creates worktree branches that can become stale if the main branch advances. Branches diverged from old commits (e.g., PR #8) would delete 25-30k lines of newer work if merged directly. These stale branches need to be deleted and work restarted on current main.
