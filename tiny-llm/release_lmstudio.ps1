param(
  [string]$BaseModelDir = "models/base_trained",
  [string]$AdapterDir = "models/lora_repair_v2",
  [string]$Checkpoint = "",
  [string]$EvalReport = "",
  [string]$ReleaseName = "tyny-lm-release2",
  [ValidateSet("Q8_0", "Q4_K_M", "F16")]
  [string]$QuantType = "Q8_0",
  [string]$LmStudioModelsRoot = "",
  [string]$LmStudioPublisher = "",
  [string]$LlamaCppRepoDir = "tools/llama.cpp",
  [string]$LlamaCppBinDir = "tools/llama.cpp-bin",
  [string]$MergeDType = "auto",
  [switch]$CleanupOldCheckpoints,
  [switch]$CleanupOldLmStudioModels
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

Push-Location $PSScriptRoot

function Resolve-LocalPath([string]$p) {
  if ([string]::IsNullOrWhiteSpace($p)) { return $p }
  if ([System.IO.Path]::IsPathRooted($p)) {
    return [System.IO.Path]::GetFullPath($p)
  }
  return [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $p))
}

function Get-CheckpointStep([string]$ckptPath) {
  if ([string]::IsNullOrWhiteSpace($ckptPath)) { return [int]::MaxValue }
  $name = Split-Path $ckptPath -Leaf
  if ($name -match '^checkpoint-(\d+)$') {
    return [int]$matches[1]
  }
  return [int]::MaxValue
}

function Remove-DirTree([string]$dirPath) {
  if (-not (Test-Path $dirPath)) { return }
  & cmd /c "rmdir /s /q `"$dirPath`""
  if ($LASTEXITCODE -ne 0) {
    throw "Failed to delete directory: $dirPath"
  }
}

function Get-LatestCheckpoint([string]$adapterDirPath) {
  $dirs = @(Get-ChildItem -Path $adapterDirPath -Directory -ErrorAction Stop |
    Where-Object { $_.Name -match '^checkpoint-\d+$' } |
    Sort-Object @{ Expression = { Get-CheckpointStep $_.FullName }; Descending = $true })
  if ($dirs.Count -eq 0) { return $null }
  return $dirs[0].FullName
}

function Get-BestCheckpointFromEvalReport([string]$reportPath) {
  if ([string]::IsNullOrWhiteSpace($reportPath)) { return $null }
  if (-not (Test-Path $reportPath)) { return $null }
  $raw = Get-Content -Path $reportPath -Raw -Encoding UTF8
  if ([string]::IsNullOrWhiteSpace($raw)) { return $null }
  $obj = $raw | ConvertFrom-Json
  if ($null -eq $obj -or $null -eq $obj.results) { return $null }

  $rows = @()
  foreach ($r in $obj.results) {
    if ($null -eq $r.checkpoint) { continue }
    $score = 0.0
    if ($null -ne $r.checkpoint_score) {
      $score = [double]$r.checkpoint_score
    }
    $rows += [PSCustomObject]@{
      checkpoint = [string]$r.checkpoint
      score = $score
      step = Get-CheckpointStep ([string]$r.checkpoint)
    }
  }

  $rows = @($rows)
  if ($rows.Count -eq 0) { return $null }

  # Conservative tie-break: highest score first, then lowest checkpoint step to reduce drift risk.
  $best = $rows |
    Sort-Object @{ Expression = { $_.score }; Descending = $true }, @{ Expression = { $_.step }; Descending = $false } |
    Select-Object -First 1
  return [string]$best.checkpoint
}

function Ensure-LlamaCppTools([string]$repoDirPath, [string]$binDirPath) {
  $convertScript = Join-Path $repoDirPath "convert_hf_to_gguf.py"
  $quantExe = Join-Path $binDirPath "llama-quantize.exe"

  if (-not (Test-Path $convertScript)) {
    Write-Host "llama.cpp repo missing, cloning..."
    New-Item -ItemType Directory -Force -Path (Split-Path $repoDirPath -Parent) | Out-Null
    & git clone --depth 1 https://github.com/ggerganov/llama.cpp $repoDirPath
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $convertScript)) {
      throw "Unable to clone llama.cpp repository into $repoDirPath"
    }
  }

  if (-not (Test-Path $quantExe)) {
    Write-Host "llama.cpp binaries missing, downloading latest Windows CPU package..."
    New-Item -ItemType Directory -Force -Path $binDirPath | Out-Null
    $headers = @{ "User-Agent" = "tiny-llm-release-script" }
    $release = Invoke-RestMethod -Uri "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest" -Headers $headers
    $asset = $release.assets | Where-Object { $_.name -like "llama-*-bin-win-cpu-x64.zip" } | Select-Object -First 1
    if ($null -eq $asset) {
      throw "Could not find a Windows CPU x64 release asset for llama.cpp."
    }
    $zipPath = Join-Path $binDirPath $asset.name
    Invoke-WebRequest -Uri $asset.browser_download_url -Headers $headers -OutFile $zipPath
    Expand-Archive -Path $zipPath -DestinationPath $binDirPath -Force
    if (-not (Test-Path $quantExe)) {
      throw "llama-quantize.exe not found after extracting $zipPath"
    }
  }

  return @{
    ConvertScript = $convertScript
    QuantExe = $quantExe
  }
}

try {
  $baseModelDirPath = Resolve-LocalPath $BaseModelDir
  $adapterDirPath = Resolve-LocalPath $AdapterDir
  if (-not (Test-Path $baseModelDirPath)) {
    throw "Base model dir not found: $baseModelDirPath"
  }
  if (-not (Test-Path $adapterDirPath)) {
    throw "Adapter dir not found: $adapterDirPath"
  }

  if ([string]::IsNullOrWhiteSpace($EvalReport)) {
    $EvalReport = Join-Path $adapterDirPath "checkpoint_eval_report.json"
  }
  $evalReportPath = Resolve-LocalPath $EvalReport

  if ([string]::IsNullOrWhiteSpace($LmStudioModelsRoot)) {
    $LmStudioModelsRoot = Join-Path $env:USERPROFILE ".lmstudio\models"
  }
  if ([string]::IsNullOrWhiteSpace($LmStudioPublisher)) {
    $LmStudioPublisher = $env:USERNAME
  }
  if ([string]::IsNullOrWhiteSpace($LmStudioPublisher)) {
    throw "Unable to resolve LM Studio publisher name. Set -LmStudioPublisher explicitly."
  }

  $selectedCheckpoint = ""
  if (-not [string]::IsNullOrWhiteSpace($Checkpoint)) {
    if ([System.IO.Path]::IsPathRooted($Checkpoint)) {
      $selectedCheckpoint = [System.IO.Path]::GetFullPath($Checkpoint)
    } else {
      $candidate = Join-Path $adapterDirPath $Checkpoint
      if (Test-Path $candidate) {
        $selectedCheckpoint = [System.IO.Path]::GetFullPath($candidate)
      } else {
        $selectedCheckpoint = Resolve-LocalPath $Checkpoint
      }
    }
  } else {
    $selectedCheckpoint = Get-BestCheckpointFromEvalReport $evalReportPath
    if ([string]::IsNullOrWhiteSpace($selectedCheckpoint)) {
      $selectedCheckpoint = Get-LatestCheckpoint $adapterDirPath
    }
  }
  if ([string]::IsNullOrWhiteSpace($selectedCheckpoint) -or -not (Test-Path $selectedCheckpoint)) {
    throw "No valid checkpoint selected. Pass -Checkpoint or provide a valid eval report."
  }

  $releaseRoot = Resolve-LocalPath (Join-Path "models/releases" $ReleaseName)
  $mergedModelDir = Join-Path $releaseRoot "merged_model"
  New-Item -ItemType Directory -Force -Path $mergedModelDir | Out-Null

  Write-Host "LM Studio release"
  Write-Host "  Base model      : $baseModelDirPath"
  Write-Host "  Adapter dir     : $adapterDirPath"
  Write-Host "  Checkpoint      : $selectedCheckpoint"
  Write-Host "  Release name    : $ReleaseName"
  Write-Host "  Quant type      : $QuantType"
  Write-Host "  Eval report     : $evalReportPath"
  Write-Host "  LM Studio root  : $LmStudioModelsRoot"
  Write-Host "  LM Studio owner : $LmStudioPublisher"
  Write-Host ""

  python .\06_merge_lora_checkpoint.py `
    --base_model_dir $baseModelDirPath `
    --adapter_checkpoint $selectedCheckpoint `
    --output_dir $mergedModelDir `
    --dtype $MergeDType
  if ($LASTEXITCODE -ne 0) {
    throw "Merge step failed."
  }

  $repoDirPath = Resolve-LocalPath $LlamaCppRepoDir
  $binDirPath = Resolve-LocalPath $LlamaCppBinDir
  $tools = Ensure-LlamaCppTools $repoDirPath $binDirPath
  $convertScript = [string]$tools.ConvertScript
  $quantExe = [string]$tools.QuantExe

  $f16Gguf = Join-Path $mergedModelDir "$ReleaseName-f16.gguf"
  Write-Host "Converting merged HF model to GGUF (F16)..."
  python $convertScript $mergedModelDir --outfile $f16Gguf --outtype f16
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $f16Gguf)) {
    throw "GGUF conversion failed."
  }

  if ($QuantType -eq "F16") {
    $finalLocalGguf = $f16Gguf
  } else {
    $quantSuffix = $QuantType.ToLowerInvariant()
    $quantGguf = Join-Path $mergedModelDir "$ReleaseName-$quantSuffix.gguf"
    Write-Host "Quantizing GGUF to $QuantType..."
    & $quantExe $f16Gguf $quantGguf $QuantType
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $quantGguf)) {
      throw "Quantization to $QuantType failed."
    }
    $finalLocalGguf = $quantGguf
  }

  $chatTemplateFile = Join-Path $releaseRoot "chat_template.jinja"
  Write-Host "Verifying GGUF chat template metadata..."
  python .\07_verify_gguf_chat_template.py `
    --gguf $finalLocalGguf `
    --reference_model_dir $baseModelDirPath `
    --write_template $chatTemplateFile
  if ($LASTEXITCODE -ne 0) {
    throw "GGUF chat template verification failed."
  }

  $publisherDir = Join-Path $LmStudioModelsRoot $LmStudioPublisher
  $lmModelDir = Join-Path $publisherDir $ReleaseName
  New-Item -ItemType Directory -Force -Path $lmModelDir | Out-Null

  $lmTargetFile = Join-Path $lmModelDir (Split-Path $finalLocalGguf -Leaf)
  try {
    Copy-Item -Force $finalLocalGguf $lmTargetFile
  } catch {
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($lmTargetFile)
    $ext = [System.IO.Path]::GetExtension($lmTargetFile)
    $ts = (Get-Date).ToString("yyyyMMdd-HHmmss")
    $fallback = Join-Path $lmModelDir "$baseName-$ts$ext"
    Copy-Item -Force $finalLocalGguf $fallback
    Write-Warning "Target file was locked. Saved fallback copy: $fallback"
    $lmTargetFile = $fallback
  }

  $info = [ordered]@{
    release_name = $ReleaseName
    base_model_dir = $baseModelDirPath
    adapter_dir = $adapterDirPath
    selected_checkpoint = $selectedCheckpoint
    eval_report = $evalReportPath
    quant_type = $QuantType
    local_gguf = $finalLocalGguf
    lmstudio_gguf = $lmTargetFile
    chat_template_file = $chatTemplateFile
    created_utc = (Get-Date).ToUniversalTime().ToString("o")
  }
  $info | ConvertTo-Json -Depth 6 | Set-Content -Path (Join-Path $releaseRoot "release_info.json") -Encoding UTF8

  if ($CleanupOldCheckpoints) {
    Write-Host "Cleaning old checkpoints in $adapterDirPath ..."
    $allCkpts = Get-ChildItem -Path $adapterDirPath -Directory | Where-Object { $_.Name -match '^checkpoint-\d+$' }
    foreach ($ckpt in $allCkpts) {
      if ([System.IO.Path]::GetFullPath($ckpt.FullName) -eq [System.IO.Path]::GetFullPath($selectedCheckpoint)) {
        continue
      }
      Remove-DirTree $ckpt.FullName
    }
  }

  if ($CleanupOldLmStudioModels) {
    Write-Host "Cleaning old LM Studio models under $publisherDir ..."
    if (Test-Path $publisherDir) {
      $allModels = Get-ChildItem -Path $publisherDir -Directory
      foreach ($m in $allModels) {
        if ($m.Name -eq $ReleaseName) { continue }
        Remove-DirTree $m.FullName
      }
    }
  }

  Write-Host ""
  Write-Host "Release completed."
  Write-Host "LM Studio model dir: $lmModelDir"
  Write-Host "GGUF: $lmTargetFile"
  Write-Host "Chat template: $chatTemplateFile"
  Write-Host "LM Studio tip: keep Prompt Template on the model default (or Empty), do not force a foreign template."
} finally {
  Pop-Location
}
