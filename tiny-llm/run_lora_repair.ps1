param(
  [string]$ModelDir = "models/base_trained",
  [string]$SeedAdapter = "models/lora_adapter_v2/best_checkpoint-300",
  [string]$OutDir = "models/lora_repair_v1",
  [int]$MaxLength = 1280,
  [int]$BatchSize = 2,
  [int]$GradAccum = 8,
  [int]$Stage1Steps = 1800,
  [double]$Stage1Lr = 8e-5,
  [switch]$DoStage2,
  [int]$Stage2ExtraSteps = 1200,
  [double]$Stage2Lr = 5e-5,
  [string]$Stage2Recipe = "standard",
  [int]$Stage2MaxRowsPerSource = 60000,
  [int]$SaveSteps = 300,
  [int]$SaveTotalLimit = 4
)

$ErrorActionPreference = "Stop"

Push-Location $PSScriptRoot

function Resolve-LocalPath([string]$p) {
  if ([string]::IsNullOrWhiteSpace($p)) { return $p }
  if ([System.IO.Path]::IsPathRooted($p)) { return $p }
  return (Join-Path $PSScriptRoot $p)
}

function Get-LatestCheckpoint([string]$dirPath) {
  if (-not (Test-Path $dirPath)) { return $null }
  $ckpt = Get-ChildItem $dirPath -Directory |
    Where-Object { $_.Name -like 'checkpoint-*' -or $_.Name -like 'interrupt-step-*' } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
  if ($null -eq $ckpt) { return $null }
  return $ckpt.FullName
}

Write-Host "LoRA Repair Run"
Write-Host "  ModelDir         : $ModelDir"
Write-Host "  SeedAdapter      : $SeedAdapter"
Write-Host "  OutDir           : $OutDir"
Write-Host "  Shape            : bs=$BatchSize, seq=$MaxLength, accum=$GradAccum"
Write-Host "  Stage1           : local-only, steps=$Stage1Steps, lr=$Stage1Lr"
Write-Host "  Stage2 enabled   : $DoStage2"
if ($DoStage2) {
  Write-Host "  Stage2           : recipe=$Stage2Recipe, extra_steps=$Stage2ExtraSteps, lr=$Stage2Lr, hf_max_rows/src=$Stage2MaxRowsPerSource"
}
Write-Host ""

$SeedAdapter = Resolve-LocalPath $SeedAdapter
if (-not (Test-Path $SeedAdapter)) {
  throw "Seed adapter checkpoint not found: $SeedAdapter"
}

$ModelDir = Resolve-LocalPath $ModelDir
$OutDir = Resolve-LocalPath $OutDir

# Stage 1: quality repair on local curated samples only.
python .\04_train_lora.py `
  --model_dir $ModelDir `
  --output_dir $OutDir `
  --resume_from_checkpoint $SeedAdapter `
  --disable_hf_data `
  --repeat_sources `
  --max_steps $Stage1Steps `
  --max_length $MaxLength `
  --per_device_batch_size $BatchSize `
  --grad_accum $GradAccum `
  --learning_rate $Stage1Lr `
  --warmup_ratio 0.03 `
  --logging_steps 20 `
  --save_steps $SaveSteps `
  --save_total_limit $SaveTotalLimit `
  --throughput_mode `
  --disable_sample_logging `
  --save_merged

if ($DoStage2) {
  $resume = Get-LatestCheckpoint $OutDir
  if ([string]::IsNullOrWhiteSpace($resume)) {
    throw "No checkpoint found after Stage 1 in $OutDir"
  }
  $stage2FinalSteps = [int]$Stage1Steps + [int]$Stage2ExtraSteps

  # Stage 2: small mixed-data pass, controlled to avoid drift.
  python .\04_train_lora.py `
    --model_dir $ModelDir `
    --output_dir $OutDir `
    --resume_from_checkpoint "$resume" `
    --recipe $Stage2Recipe `
    --max_rows_per_source $Stage2MaxRowsPerSource `
    --repeat_sources `
    --max_steps $stage2FinalSteps `
    --max_length $MaxLength `
    --per_device_batch_size $BatchSize `
    --grad_accum $GradAccum `
    --learning_rate $Stage2Lr `
    --warmup_ratio 0.03 `
    --logging_steps 20 `
    --save_steps $SaveSteps `
    --save_total_limit $SaveTotalLimit `
    --throughput_mode `
    --disable_sample_logging `
    --save_merged
}

Write-Host ""
Write-Host "Repair run complete."
Write-Host "Output dir: $OutDir"
Write-Host "Next eval:"
Write-Host "python .\05_eval_lora_checkpoints.py --base_model_dir $ModelDir --adapter_dir $OutDir --max_checkpoints 4"

Pop-Location
