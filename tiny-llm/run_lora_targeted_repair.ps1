param(
  [string]$ModelDir = "models/base_trained",
  [string]$SeedAdapter = "models/lora_repair_v1/checkpoint-900",
  [string]$OutDir = "models/lora_repair_v2",
  [int]$MaxLength = 1280,
  [int]$BatchSize = 2,
  [int]$GradAccum = 8,
  [int]$MaxSteps = 1400,
  [double]$LearningRate = 6e-5,
  [int]$SaveSteps = 200,
  [int]$SaveTotalLimit = 6
)

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot

function Resolve-LocalPath([string]$p) {
  if ([string]::IsNullOrWhiteSpace($p)) { return $p }
  if ([System.IO.Path]::IsPathRooted($p)) { return $p }
  return (Join-Path $PSScriptRoot $p)
}

function Get-CheckpointStep([string]$ckptPath) {
  if (-not (Test-Path $ckptPath)) { return 0 }
  $stateFile = Join-Path $ckptPath "trainer_state.json"
  if (Test-Path $stateFile) {
    try {
      $state = Get-Content $stateFile -Raw | ConvertFrom-Json
      if ($null -ne $state.global_step) {
        return [int]$state.global_step
      }
    } catch {
      # Fallback to checkpoint directory name parsing.
    }
  }
  $name = Split-Path $ckptPath -Leaf
  if ($name -match '^checkpoint-(\d+)$') {
    return [int]$matches[1]
  }
  return 0
}

Write-Host "LoRA Targeted Repair Run"
Write-Host "  ModelDir     : $ModelDir"
Write-Host "  SeedAdapter  : $SeedAdapter"
Write-Host "  OutDir       : $OutDir"
Write-Host "  Shape        : bs=$BatchSize, seq=$MaxLength, accum=$GradAccum"
Write-Host "  Extra/LR     : extra_steps=$MaxSteps, lr=$LearningRate"
Write-Host ""

$ModelDir = Resolve-LocalPath $ModelDir
$SeedAdapter = Resolve-LocalPath $SeedAdapter
$OutDir = Resolve-LocalPath $OutDir

if (-not (Test-Path $ModelDir)) {
  throw "Model dir not found: $ModelDir"
}
if (-not (Test-Path $SeedAdapter)) {
  throw "Seed adapter checkpoint not found: $SeedAdapter"
}

$resumeStep = Get-CheckpointStep $SeedAdapter
$targetMaxSteps = [int]$resumeStep + [int]$MaxSteps
Write-Host "  Resume step  : $resumeStep"
Write-Host "  Target step  : $targetMaxSteps"
Write-Host ""

python .\04_train_lora.py `
  --model_dir $ModelDir `
  --output_dir $OutDir `
  --resume_from_checkpoint $SeedAdapter `
  --disable_hf_data `
  --repeat_sources `
  --local_jsonl_glob "samples/sft/repair_math_logic_coding.jsonl" `
  --local_jsonl_glob "samples/sft/system_styles.jsonl" `
  --local_jsonl_glob "samples/sft/chat_alignment_samples.jsonl" `
  --max_steps $targetMaxSteps `
  --max_length $MaxLength `
  --per_device_batch_size $BatchSize `
  --grad_accum $GradAccum `
  --learning_rate $LearningRate `
  --warmup_ratio 0.03 `
  --logging_steps 20 `
  --save_steps $SaveSteps `
  --save_total_limit $SaveTotalLimit `
  --throughput_mode `
  --disable_sample_logging `
  --save_merged

Write-Host ""
Write-Host "Targeted repair complete."
Write-Host "Output dir: $OutDir"
Write-Host "Next eval:"
Write-Host "python .\05_eval_lora_checkpoints.py --base_model_dir $ModelDir --adapter_dir $OutDir --max_checkpoints 6"

Pop-Location
