param(
  [string]$ModelId = "Qwen/Qwen2.5-3B-Instruct",
  [string]$BaseDir = "models/base_3b",
  [string]$CptDir = "models/base_3b_code_fast_16gb_v1",
  [string]$LoraDir = "models/lora3b_code_review_seed_v1",
  [string]$CodeEvalPrompts = "samples/eval/code_assistant_eval.jsonl",
  [string]$SyntheticSftPath = "samples/sft/code_review_synthetic.jsonl",
  [int]$SyntheticSftCount = 320,
  [int]$CptSteps = 6000,
  [int]$SftSteps = 400,
  [switch]$RefreshSyntheticSft,
  [switch]$SkipDownloadBase,
  [switch]$SkipCpt,
  [switch]$SkipSft
)

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot

function Resolve-LocalPath([string]$p) {
  if ([string]::IsNullOrWhiteSpace($p)) { return $p }
  if ([System.IO.Path]::IsPathRooted($p)) { return $p }
  return (Join-Path $PSScriptRoot $p)
}

$BaseDir = Resolve-LocalPath $BaseDir
$CptDir = Resolve-LocalPath $CptDir
$LoraDir = Resolve-LocalPath $LoraDir
$CodeEvalPrompts = Resolve-LocalPath $CodeEvalPrompts
$SyntheticSftPath = Resolve-LocalPath $SyntheticSftPath

Write-Host "3B Programming + Code Review Pipeline"
Write-Host "  ModelId: $ModelId"
Write-Host "  BaseDir: $BaseDir"
Write-Host "  CPTDir : $CptDir"
Write-Host "  LoRADir: $LoraDir"
Write-Host ""

if (-not $SkipDownloadBase) {
  & python .\01_download_base.py `
    --model_id "$ModelId" `
    --output_dir "$BaseDir" `
    --dtype float16
  if ($LASTEXITCODE -ne 0) { throw "Base download failed." }
}

if (-not $SkipCpt) {
  & python .\02_train_base.py `
    --model_dir "$BaseDir" `
    --output_dir "$CptDir" `
    --disable_local_data `
    --hf_source "codeparrot/github-code||train|code|800000" `
    --hf_code_languages "python,typescript" `
    --hf_require_language_tag `
    --repeat_sources `
    --max_steps "$CptSteps" `
    --learning_rate 2e-5 `
    --warmup_ratio 0.02 `
    --per_device_batch_size 2 `
    --grad_accum 12 `
    --block_size 768 `
    --auto_tune_shape `
    --auto_tune_batch_candidates "1,2,3" `
    --auto_tune_block_candidates "512,768,1024" `
    --gradient_checkpointing `
    --dtype float16 `
    --logging_steps 20 `
    --save_steps 500 `
    --save_total_limit 4 `
    --disable_sample_logging
  if ($LASTEXITCODE -ne 0) { throw "Base CPT failed." }
}

if (-not $SkipSft) {
  $regenSynthetic = $false
  if ($RefreshSyntheticSft) {
    $regenSynthetic = $true
  } elseif (-not (Test-Path $SyntheticSftPath)) {
    $regenSynthetic = $true
  } else {
    $lineCount = (Get-Content $SyntheticSftPath | Measure-Object -Line).Lines
    if ([int]$lineCount -lt [int]$SyntheticSftCount) {
      $regenSynthetic = $true
    }
  }

  if ($regenSynthetic) {
    Write-Host "Generating synthetic SFT dataset: $SyntheticSftPath"
    & python .\tools\generate_code_review_dataset.py `
      --out_jsonl "$SyntheticSftPath" `
      --count "$SyntheticSftCount"
    if ($LASTEXITCODE -ne 0) { throw "Synthetic SFT dataset generation failed." }
  }

  & python .\04_train_lora.py `
    --model_dir "$CptDir" `
    --output_dir "$LoraDir" `
    --disable_hf_data `
    --validate_data `
    --chat_format tokenizer `
    --code_fence_hygiene normalize `
    --reject_no_markdown_code_examples `
    --fail_on_duplicate_examples `
    --max_duplicate_example_ratio 0.10 `
    --min_loaded_examples 200 `
    --local_jsonl_glob "samples/sft/code_review_seed.jsonl" `
    --local_jsonl_glob "samples/sft/code_assistant_booster.jsonl" `
    --local_jsonl_glob "samples/sft/code_review_synthetic.jsonl" `
    --local_jsonl_glob "samples/sft/repair_math_logic_coding.jsonl" `
    --local_jsonl_glob "samples/sft/system_styles.jsonl" `
    --local_jsonl_glob "samples/sft/chat_alignment_samples.jsonl" `
    --local_jsonl_glob "samples/sft/formatting_code_fences.jsonl" `
    --local_jsonl_glob "samples/sft/format_constraints_strict.jsonl" `
    --local_jsonl_glob "samples/sft/math_reasoning_micro.jsonl" `
    --max_steps "$SftSteps" `
    --max_length 1024 `
    --per_device_batch_size 1 `
    --grad_accum 16 `
    --learning_rate 4e-5 `
    --gradient_checkpointing `
    --dtype float16 `
    --logging_steps 20 `
    --save_steps 100 `
    --save_total_limit 6 `
    --throughput_mode `
    --disable_sample_logging
  if ($LASTEXITCODE -ne 0) { throw "LoRA SFT failed." }

  & python .\05_eval_lora_checkpoints.py `
    --base_model_dir "$CptDir" `
    --adapter_dir "$LoraDir" `
    --max_checkpoints 6 `
    --out_json "$LoraDir/checkpoint_eval_report.json"
  if ($LASTEXITCODE -ne 0) { throw "Checkpoint eval failed." }

  $reportPath = Join-Path $LoraDir "checkpoint_eval_report.json"
  if (Test-Path $reportPath) {
    $report = Get-Content $reportPath -Raw | ConvertFrom-Json
    $best = $report.results | Sort-Object checkpoint_score -Descending | Select-Object -First 1
    if ($null -ne $best) {
      Write-Host "Best checkpoint from 05_eval_lora_checkpoints.py: $($best.checkpoint)"
      & python .\08_eval_code_assistant.py `
        --base_model_dir "$CptDir" `
        --adapter_dir "$($best.checkpoint)" `
        --prompts_jsonl "$CodeEvalPrompts" `
        --out_json "$LoraDir/code_assistant_eval_report.json" `
        --temperature 0.0 `
        --max_new_tokens 240
      if ($LASTEXITCODE -ne 0) { throw "Code-assistant eval failed." }
    }
  }
}

Write-Host ""
Write-Host "Pipeline complete."
Write-Host "Next:"
Write-Host "  1) Inspect: $LoraDir/checkpoint_eval_report.json"
Write-Host "  2) Inspect: $LoraDir/code_assistant_eval_report.json"
Write-Host "  3) Promote best checkpoint and merge with 06_merge_lora_checkpoint.py or release_lmstudio.ps1"

Pop-Location
