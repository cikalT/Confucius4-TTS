# Pre-flight checks for 1_setup.bat: Windows version, Intel Arc GPU + driver, free disk space.
# Exit code: 0 = OK, 2 = warnings (setup can continue), 1 = cannot continue.
param([string]$RepoDir = (Get-Location).Path)

$warn = $false
$os = Get-CimInstance Win32_OperatingSystem
Write-Host ("  Windows : {0} (build {1})" -f $os.Caption, $os.BuildNumber)
if (-not [Environment]::Is64BitOperatingSystem) {
    Write-Host "  [GAGAL] Butuh Windows 64-bit." -ForegroundColor Red
    exit 1
}

$gpus = Get-CimInstance Win32_VideoController
$arc = $gpus | Where-Object { $_.Name -match 'Intel' -and $_.Name -match 'Arc' }
if ($arc) {
    foreach ($g in $arc) { Write-Host ("  GPU     : {0} (driver {1})" -f $g.Name, $g.DriverVersion) }
} else {
    $names = ($gpus | ForEach-Object { $_.Name }) -join ', '
    Write-Host "  [PERINGATAN] GPU Intel Arc tidak ditemukan (terdeteksi: $names)." -ForegroundColor Yellow
    Write-Host "               Setup tetap bisa jalan, tapi model akan sangat lambat tanpa GPU Arc." -ForegroundColor Yellow
    $warn = $true
}

$ramGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
Write-Host ("  RAM     : {0} GB" -f $ramGB)
if ($ramGB -lt 16) {
    Write-Host "  [PERINGATAN] RAM kurang dari 16 GB; memuat model bisa lambat." -ForegroundColor Yellow
    $warn = $true
}

# .venv (~5 GB) goes next to the repo; models (~5 GB) and the uv cache (~3 GB) go to the user profile.
$checks = @(
    @{ Label = 'Folder proyek'; Path = $RepoDir; NeedGB = 8 },
    @{ Label = 'Folder user (model + cache)'; Path = $env:USERPROFILE; NeedGB = 10 }
)
foreach ($c in $checks) {
    $drive = (Get-Item $c.Path).PSDrive
    $freeGB = [math]::Round($drive.Free / 1GB, 1)
    Write-Host ("  Disk {0}: bebas {1} GB ({2}, butuh +-{3} GB)" -f $drive.Name, $freeGB, $c.Label, $c.NeedGB)
    if ($freeGB -lt $c.NeedGB) {
        Write-Host ("  [GAGAL] Ruang disk {0}: tidak cukup." -f $drive.Name) -ForegroundColor Red
        exit 1
    }
}

try {
    $null = Invoke-WebRequest -UseBasicParsing -Uri 'https://huggingface.co' -Method Head -TimeoutSec 15
    Write-Host "  Internet: OK (huggingface.co)"
} catch {
    Write-Host "  [GAGAL] Tidak bisa menghubungi huggingface.co. Cek koneksi internet / proxy." -ForegroundColor Red
    exit 1
}

if ($warn) { exit 2 } else { exit 0 }
