param(
  [string]$ModelDir = "models/base_trained",
  [string]$SeedAdapter = "models/lora_repair_v1/checkpoint-900",
  [string]$OutDir = "models/lora_repair_v2",
  [int]$MaxLength = 1280,
  [int]$BatchSize = 2,
  [int]$GradAccum = 8,
  [int]$MaxSteps = 120,
  [double]$LearningRate = 8e-6,
  [ValidateSet("auto", "float16", "bfloat16", "float32")]
  [string]$DType = "float16",
  [switch]$Use4Bit,
  [ValidateSet("nf4", "fp4")]
  [string]$Bnb4BitQuantType = "nf4",
  [ValidateSet("auto", "float16", "bfloat16", "float32")]
  [string]$Bnb4BitComputeDType = "auto",
  [switch]$Disable4BitDoubleQuant,
  [int]$SaveSteps = 60,
  [int]$SaveTotalLimit = 6,
  [int]$MinLoadedExamples = 250,
  [double]$MaxDuplicateExampleRatio = 0.10,
  [switch]$RepeatSources,
  [switch]$AllowExistingOutDir
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
Write-Host "  DType        : $DType"
if ($Use4Bit) {
  $doubleQuantState = if ($Disable4BitDoubleQuant) { "off" } else { "on" }
  Write-Host "  QLoRA 4-bit  : quant=$Bnb4BitQuantType, compute_dtype=$Bnb4BitComputeDType, double_quant=$doubleQuantState"
}
Write-Host "  Data guard   : strict JSONL validation + code-fence hygiene"
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
if ((Test-Path $OutDir) -and (-not $AllowExistingOutDir)) {
  $existingCkpts = @(Get-ChildItem -Path $OutDir -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^checkpoint-\d+$' })
  if ($existingCkpts.Count -gt 0) {
    throw "Output dir already contains checkpoints: $OutDir. Use a fresh -OutDir or pass -AllowExistingOutDir."
  }
}

$resumeStep = Get-CheckpointStep $SeedAdapter
$targetMaxSteps = [int]$resumeStep + [int]$MaxSteps
Write-Host "  Resume step  : $resumeStep"
Write-Host "  Target step  : $targetMaxSteps"
Write-Host "  Repeat source: $($RepeatSources.IsPresent)"
Write-Host ""

$trainArgs = @(
  "--model_dir", $ModelDir,
  "--output_dir", $OutDir,
  "--resume_from_checkpoint", $SeedAdapter,
  "--validate_data",
  "--disable_hf_data",
  "--chat_format", "tokenizer",
  "--code_fence_hygiene", "normalize",
  "--reject_no_markdown_code_examples",
  "--fail_on_duplicate_examples",
  "--max_duplicate_example_ratio", "$MaxDuplicateExampleRatio",
  "--min_loaded_examples", "$MinLoadedExamples",
  "--local_jsonl_glob", "samples/sft/repair_math_logic_coding.jsonl",
  "--local_jsonl_glob", "samples/sft/code_review_seed.jsonl",
  "--local_jsonl_glob", "samples/sft/code_assistant_booster.jsonl",
  "--local_jsonl_glob", "samples/sft/code_review_synthetic.jsonl",
  "--local_jsonl_glob", "samples/sft/system_styles.jsonl",
  "--local_jsonl_glob", "samples/sft/chat_alignment_samples.jsonl",
  "--local_jsonl_glob", "samples/sft/formatting_code_fences.jsonl",
  "--local_jsonl_glob", "samples/sft/format_constraints_strict.jsonl",
  "--local_jsonl_glob", "samples/sft/math_reasoning_micro.jsonl",
  "--max_steps", "$targetMaxSteps",
  "--dtype", "$DType",
  "--max_length", "$MaxLength",
  "--per_device_batch_size", "$BatchSize",
  "--grad_accum", "$GradAccum",
  "--learning_rate", "$LearningRate",
  "--warmup_ratio", "0.03",
  "--logging_steps", "20",
  "--save_steps", "$SaveSteps",
  "--save_total_limit", "$SaveTotalLimit",
  "--throughput_mode",
  "--disable_sample_logging",
  "--save_merged"
)

if ($RepeatSources) {
  $trainArgs += "--repeat_sources"
}
if ($Use4Bit) {
  $trainArgs += @(
    "--use_4bit",
    "--bnb_4bit_quant_type", "$Bnb4BitQuantType",
    "--bnb_4bit_compute_dtype", "$Bnb4BitComputeDType"
  )
  if ($Disable4BitDoubleQuant) {
    $trainArgs += "--no_bnb_4bit_use_double_quant"
  }
}

& python .\04_train_lora.py @trainArgs
if ($LASTEXITCODE -ne 0) {
  throw "Targeted repair training failed."
}

Write-Host ""
Write-Host "Targeted repair complete."
Write-Host "Output dir: $OutDir"
Write-Host "Next eval:"
Write-Host "python .\05_eval_lora_checkpoints.py --base_model_dir $ModelDir --adapter_dir $OutDir --max_checkpoints 6"

Pop-Location
