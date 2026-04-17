# Mac Installer via GitHub Actions (No Mac Required)

This guide builds a macOS installer in GitHub's cloud, even if you only use Windows.

## 1) Push this project to GitHub

1. Create a GitHub repository (or use your existing one).
2. Push your current code (including `.github/workflows/build-mac.yml`).

## 2) Run the workflow manually

1. Open your repository on GitHub.
2. Go to **Actions**.
3. Select **Build macOS Installer**.
4. Click **Run workflow**.
5. Wait for completion (usually 5-15 minutes).

## 3) Download the macOS installer

1. Open the finished workflow run.
2. In **Artifacts**, download `trading-journal-macos`.
3. Extract it and you will get files like:
   - `Trading Journal-1.0.0.dmg`
   - optional `latest-mac.yml` / `.zip`

## 4) Deliver to your Mac user

Share these two parts:

1. The `.dmg` file (installer)
2. Customer `license` folder with:
   - customer license file (for example `jack.lic.json`)
   - `public_key.pem`

## 5) Mac user install instructions

1. Open `.dmg` and drag app to Applications.
2. Place the `license` folder beside the app executable folder if your runtime requires it.
3. First launch (unsigned app): right-click app -> Open.

## Notes

- Unsigned builds can trigger Gatekeeper warnings on macOS.
- For smoother distribution, add Apple code signing + notarization later.
- This workflow currently builds on `main`/`master` pushes and manual trigger.
