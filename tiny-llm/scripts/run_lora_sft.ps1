param(
  [string]$ModelDir = "models/base_trained",
  [string]$OutDir = "models/lora_adapter",
  [string]$Recipe = "heavy",
  [int]$MaxSteps = 12000,
  [int]$MaxLength = 1280,
  [int]$BatchSize = 2,
  [int]$GradAccum = 8
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot

Write-Host "LoRA SFT run"
Write-Host "  ModelDir : $ModelDir"
Write-Host "  OutDir   : $OutDir"
Write-Host "  Recipe   : $Recipe"
Write-Host "  Steps    : $MaxSteps"
Write-Host "  Length   : $MaxLength"
Write-Host "  BS       : $BatchSize"
Write-Host "  Accum    : $GradAccum"
Write-Host ""

python .\04_train_lora.py `
  --model_dir $ModelDir `
  --output_dir $OutDir `
  --recipe $Recipe `
  --repeat_sources `
  --max_steps $MaxSteps `
  --max_length $MaxLength `
  --per_device_batch_size $BatchSize `
  --grad_accum $GradAccum `
  --learning_rate 2e-4 `
  --warmup_ratio 0.05 `
  --logging_steps 20 `
  --save_steps 300 `
  --save_total_limit 4 `
  --throughput_mode `
  --save_merged `
  --sample_log_steps 200 `
  --sample_log_count 2 `
  --sample_preview_per_source 0 `
  --sample_eval_prompt "User: Scrivi un messaggio di benvenuto in italiano, amichevole ma professionale.`n`nAssistant:" `
  --sample_eval_prompt "User: Write a short checklist (5 items) for debugging a Python bug.`n`nAssistant:" `
  --sample_eval_prompt "User: Risolvi: 18*7. Rispondi con il numero e una breve spiegazione.`n`nAssistant:" `
  --sample_eval_prompt "User: Se tutti i gatti sono mammiferi e tutti i mammiferi respirano, i gatti respirano? Rispondi si/no e 1 riga.`n`nAssistant:"

Pop-Location
