# Dubber installer: Python env + libraries + models + desktop shortcut.  No admin rights needed.
param([switch]$NoShortcut)
$ErrorActionPreference = 'Stop'
$OutputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Step($t) { Write-Host ""; Write-Host "=== $t ===" -ForegroundColor Cyan }
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "Ошибка на шаге: $what" } }

Step "1/5  Проверяю менеджер Python (uv)"
$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    Write-Host "Ставлю uv (официальный установщик Astral, ~20 МБ)..."
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uv) { throw "Не удалось установить uv. Установите вручную: https://docs.astral.sh/uv/" }
}
Write-Host "uv найден: $($uv.Source)"

Step "2/5  Python 3.11 и изолированное окружение"
uv venv --python 3.11 --allow-existing server\.venv
Check "uv venv"
$py = Join-Path $root "server\.venv\Scripts\python.exe"

Step "3/5  Библиотеки (около 1,5 ГБ)"
uv pip install --python $py torch --index-url https://download.pytorch.org/whl/cpu
Check "torch"
uv pip install --python $py -r server\requirements.txt
Check "requirements"

Step "4/5  Модели: распознавание, перевод, голос (около 3 ГБ, один раз)"
$env:PYTHONUTF8 = "1"
Push-Location server
& $py download_models.py
$modelsOk = ($LASTEXITCODE -eq 0)
Pop-Location
if (-not $modelsOk) { Write-Host "Модели скачались не полностью. Запустите install.bat ещё раз." -ForegroundColor Yellow }

Step "5/5  Ярлык на рабочем столе"
if ($NoShortcut) { Write-Host "(ярлык пропущен)" } else { try {
    $desktop = [Environment]::GetFolderPath('Desktop')
    $ws = New-Object -ComObject WScript.Shell
    $lnk = $ws.CreateShortcut((Join-Path $desktop "Dubber.lnk"))
    $lnk.TargetPath = Join-Path $root "start.bat"
    $lnk.WorkingDirectory = $root
    $lnk.Description = "Dubber - озвучка видео на русском"
    $lnk.Save()
    Write-Host "Ярлык «Dubber» создан на рабочем столе."
} catch { Write-Host "Ярлык создать не удалось - запускайте start.bat вручную." -ForegroundColor Yellow } }

$ext = Join-Path $root "extension"
Write-Host ""
Write-Host "=====================================================" -ForegroundColor Green
Write-Host " Установка завершена." -ForegroundColor Green
Write-Host "=====================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Осталось поставить расширение в Chrome (один раз):"
Write-Host "  1. Откройте chrome://extensions"
Write-Host "  2. Включите «Режим разработчика» (справа сверху)"
Write-Host "  3. Нажмите «Загрузить распакованное расширение» и выберите папку:"
Write-Host "       $ext" -ForegroundColor Yellow
Write-Host ""
Write-Host "Использование: запустите ярлык «Dubber», откройте видео, нажмите значок расширения -> «Озвучить»."
