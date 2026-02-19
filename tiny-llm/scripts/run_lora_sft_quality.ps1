param(
  [string]$ModelDir = "models/base_trained",
  [string]$OutDir = "models/lora_adapter_v2",
  [int]$MaxLength = 1280,
  [int]$BatchSize = 2,
  [int]$GradAccum = 8,
  [int]$Stage1Steps = 1800,
  [int]$Stage2Steps = 5200,
  [double]$Stage1Lr = 1e-4,
  [double]$Stage2Lr = 8e-5,
  [string]$Stage2Recipe = "standard",
  [int]$HfMaxRowsPerSource = 120000
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot

function Get-LatestCheckpoint([string]$dirPath) {
  if (-not (Test-Path $dirPath)) { return $null }
  $ckpt = Get-ChildItem $dirPath -Directory |
    Where-Object { $_.Name -like 'checkpoint-*' -or $_.Name -like 'interrupt-step-*' } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
  if ($null -eq $ckpt) { return $null }
  return $ckpt.FullName
}

Write-Host "LoRA SFT Quality Run"
Write-Host "  ModelDir         : $ModelDir"
Write-Host "  OutDir           : $OutDir"
Write-Host "  Shape            : bs=$BatchSize, seq=$MaxLength, accum=$GradAccum"
Write-Host "  Stage1 (local)   : steps=$Stage1Steps, lr=$Stage1Lr"
Write-Host "  Stage2 (mixed)   : steps=$Stage2Steps, lr=$Stage2Lr, recipe=$Stage2Recipe"
Write-Host "  HF max rows/src  : $HfMaxRowsPerSource"
Write-Host ""

# Stage 1: local-only quality alignment (stabilize style/format)
python .\04_train_lora.py `
  --model_dir $ModelDir `
  --output_dir $OutDir `
  --disable_hf_data `
  --repeat_sources `
  --max_steps $Stage1Steps `
  --max_length $MaxLength `
  --per_device_batch_size $BatchSize `
  --grad_accum $GradAccum `
  --learning_rate $Stage1Lr `
  --warmup_ratio 0.03 `
  --logging_steps 20 `
  --save_steps 300 `
  --save_total_limit 12 `
  --throughput_mode `
  --torch_compile `
  --torch_compile_backend aot_eager `
  --attn_implementation sdpa `
  --disable_sample_logging

$resume = Get-LatestCheckpoint $OutDir
if ([string]::IsNullOrWhiteSpace($resume)) {
  throw "No checkpoint found after Stage 1 in $OutDir"
}

$totalSteps = $Stage1Steps + $Stage2Steps

# Stage 2: controlled mixed data (generalize without drifting style)
python .\04_train_lora.py `
  --model_dir $ModelDir `
  --output_dir $OutDir `
  --recipe $Stage2Recipe `
  --max_rows_per_source $HfMaxRowsPerSource `
  --repeat_sources `
  --max_steps $totalSteps `
  --max_length $MaxLength `
  --per_device_batch_size $BatchSize `
  --grad_accum $GradAccum `
  --learning_rate $Stage2Lr `
  --warmup_ratio 0.03 `
  --logging_steps 20 `
  --save_steps 300 `
  --save_total_limit 12 `
  --throughput_mode `
  --torch_compile `
  --torch_compile_backend aot_eager `
  --attn_implementation sdpa `
  --resume_from_checkpoint "$resume" `
  --disable_sample_logging `
  --save_merged `
  --sample_eval_prompt "User: Write a short welcome message in Italian for a new user (2 sentences).`n`nAssistant:" `
  --sample_eval_prompt "User: Write a short checklist (5 items) for debugging a Python bug.`n`nAssistant:" `
  --sample_eval_prompt "User: Solve 18*7. Return number plus one short sentence.`n`nAssistant:" `
  --sample_eval_prompt "User: If all cats are mammals and all mammals breathe, do cats breathe? Answer yes/no plus one line.`n`nAssistant:"

Write-Host ""
Write-Host "Quality run complete."
Write-Host "Merged model (if successful): $OutDir\\merged_model"

Pop-Location
