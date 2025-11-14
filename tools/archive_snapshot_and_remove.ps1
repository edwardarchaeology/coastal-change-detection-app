<#
Archive snapshot and prepare cleanup branch (PowerShell)

Usage (run in repository root PowerShell):
    .\tools\archive_snapshot_and_remove.ps1

What this does:
1. Creates a timestamped zip of the `archive/` folder one level above the repo (so it's not lost in git).
2. Creates a new git branch `repo-cleanup` and checks it out.
3. Removes `archive/` from the repository (git rm -r) and commits the change.

Note: This script does NOT push the branch for you. Review the generated zip in the parent folder before pushing.
#>

Write-Host "Preparing archive snapshot and cleanup branch..."

$timestamp = Get-Date -Format "yyyyMMdd_HHmm"
$parent = Split-Path -Parent $PWD
$zipName = "archive_snapshot_$timestamp.zip"
$zipPath = Join-Path $parent $zipName

if (-not (Test-Path -Path "archive")) {
    Write-Host "No 'archive' folder found in repository root. Aborting." -ForegroundColor Yellow
    exit 1
}

Write-Host "Creating zip snapshot at: $zipPath"
Compress-Archive -Path "archive\*" -DestinationPath $zipPath -Force

if ($LASTEXITCODE -ne 0) {
    Write-Host "Compress-Archive returned an error. Check permissions and try again." -ForegroundColor Red
    exit 1
}

Write-Host "Snapshot created. Now creating git branch 'repo-cleanup' and removing archive/ from the repo."

git checkout -b repo-cleanup
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to create branch. Maybe branch already exists. Please check and run manually." -ForegroundColor Yellow
}

git rm -r archive
if ($LASTEXITCODE -ne 0) {
    Write-Host "git rm failed. You may need to remove archive/ manually or check git status." -ForegroundColor Yellow
}

git commit -m "repo-cleanup: remove archived versions from main tree (snapshot: $zipName)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "git commit failed. Please inspect `git status` and commit manually." -ForegroundColor Yellow
} else {
    Write-Host "Committed removal of archive/. Review changes and push the branch when ready: git push -u origin repo-cleanup"
}

Write-Host "Done. Snapshot stored at: $zipPath"
