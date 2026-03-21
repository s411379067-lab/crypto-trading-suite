# Crypto App Monorepo Versioning Strategy

This strategy is designed for a single repository containing 3 apps:
- backtest app
- regime analyze app
- trading journal app

## 1) Branch Naming

Default branches:
- `main`: production-ready code only
- `develop`: optional integration branch (recommended when changes are frequent)

Feature and maintenance branches:
- `feature/<app>/<short-topic>`
- `fix/<app>/<short-topic>`
- `refactor/<app>/<short-topic>`
- `chore/<scope>/<short-topic>`
- `release/<app>/v<major>.<minor>.<patch>`
- `hotfix/<app>/<short-topic>`

Allowed `<app>` values:
- `backtest`
- `regime`
- `journal`
- `shared` (for cross-app shared logic)

Examples:
- `feature/backtest/add-equity-curve`
- `fix/regime/handle-empty-signal`
- `release/journal/v1.3.0`
- `hotfix/shared/fix-timezone-parse`

## 2) Commit Convention (Conventional Commits)

Format:
- `<type>(<scope>): <summary>`

Types:
- `feat`: new feature
- `fix`: bug fix
- `refactor`: code refactor without behavior change
- `perf`: performance improvement
- `docs`: docs-only changes
- `test`: tests added/updated
- `build`: build or packaging changes
- `ci`: CI/CD pipeline changes
- `chore`: maintenance changes

Scopes (recommended):
- `backtest`
- `regime`
- `journal`
- `shared`
- `release`

Examples:
- `feat(backtest): add slippage setting in UI`
- `fix(regime): avoid crash when candle data is empty`
- `build(journal): update electron builder target to nsis`
- `chore(release): prepare journal v1.2.1`

Breaking changes:
- Add `!` after type/scope, and explain in body.
- Example: `feat(shared)!: replace signal schema v1 with v2`

## 3) Versioning Rule (SemVer)

Use Semantic Versioning: `MAJOR.MINOR.PATCH`
- MAJOR: breaking change
- MINOR: backward-compatible feature
- PATCH: backward-compatible bug fix

Each app versions independently.

Recommended location for app version source of truth:
- `backtest app/VERSION`
- `regime analyze app/VERSION`
- `trading journal app/package.json` (already has `version`)

## 4) Tag Rule

Tag format:
- `<app>-v<major>.<minor>.<patch>`

Tag app names:
- `backtest-ui`
- `regime-analyzer`
- `trading-journal`

Examples:
- `backtest-ui-v1.0.0`
- `regime-analyzer-v0.4.2`
- `trading-journal-v1.2.0`

Create and push tag:

```bash
git tag backtest-ui-v1.0.0
git push origin backtest-ui-v1.0.0
```

## 5) Release Flow (Per App)

1. Create working branch
- `feature/<app>/<topic>` or `fix/<app>/<topic>`

2. Open PR into `main` (or `develop` then merge to `main`)
- Use PR template and checklist

3. Bump app version
- Update `VERSION` file or `package.json` in target app

4. Build Windows app package
- Generate `.exe` installer and checksum if possible

5. Create git tag with app prefix
- Example: `trading-journal-v1.2.0`

6. Publish GitHub Release
- Use release template in `.github/RELEASE_TEMPLATE.md`
- Attach installer and checksum files

## 6) Pull Request Rules

Minimum PR requirements:
- Clear title following commit style (`type(scope): summary`)
- Include app impact: backtest/regime/journal/shared
- Include test result or manual verification steps
- Include screenshot/video for UI changes
- Include rollback note for risky changes

## 7) Fast Start Commands

Set commit template:

```bash
git config commit.template .gitmessage
```

Create a branch example:

```bash
git checkout -b feature/backtest/add-trade-filter
```

Commit example:

```bash
git add .
git commit -m "feat(backtest): add trade filter by volatility"
```

Release example (journal):

```bash
git checkout main
git pull
git tag trading-journal-v1.2.0
git push origin trading-journal-v1.2.0
```
