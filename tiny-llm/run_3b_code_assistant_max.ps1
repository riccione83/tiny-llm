param(
  [string]$ModelId = "Qwen/Qwen2.5-3B-Instruct",
  [string]$BaseDir = "models/base_3b",
  [string]$CptDir = "models/base_3b_code_max_16gb_v1",
  [string]$SeedDir = "models/lora3b_code_review_max_seed_v1",
  [string]$RepairDir = "models/lora3b_code_review_max_repair_v1",
  [string]$CodeEvalPrompts = "samples/eval/code_assistant_eval.jsonl",
  [string]$SyntheticSftPath = "samples/sft/code_review_synthetic.jsonl",
  [int]$SyntheticSftCount = 480,
  [int]$CptSteps = 12000,
  [int]$SeedSteps = 800,
  [int]$RepairExtraSteps = 180,
  [int]$EvalTopK = 3,
  [switch]$RefreshSyntheticSft,
  [switch]$SkipDownloadBase,
  [switch]$SkipCpt,
  [switch]$SkipSeed,
  [switch]$SkipRepair
)

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot

# Keep runtime conservative on 16GB VRAM:
# - disable dynamo graph compilation (can spike memory)
# - enable expandable segments allocator to reduce fragmentation
$env:TORCHDYNAMO_DISABLE = "1"
if ([string]::IsNullOrWhiteSpace($env:PYTORCH_ALLOC_CONF)) {
  $env:PYTORCH_ALLOC_CONF = "expandable_segments:True"
}

function Resolve-LocalPath([string]$p) {
  if ([string]::IsNullOrWhiteSpace($p)) { return $p }
  if ([System.IO.Path]::IsPathRooted($p)) { return $p }
  return (Join-Path $PSScriptRoot $p)
}

function Get-TopCheckpointsFromEvalReport([string]$reportPath, [int]$topK) {
  if (-not (Test-Path $reportPath)) { return @() }
  $rep = Get-Content $reportPath -Raw | ConvertFrom-Json
  if ($null -eq $rep -or $null -eq $rep.results) { return @() }
  $sorted = $rep.results | Sort-Object checkpoint_score -Descending | Select-Object -First ([Math]::Max(1, $topK))
  $paths = @()
  foreach ($r in $sorted) {
    if ($null -ne $r.checkpoint -and -not [string]::IsNullOrWhiteSpace([string]$r.checkpoint)) {
      $paths += [string]$r.checkpoint
    }
  }
  return $paths
}

function Invoke-CodeEval([string]$baseDir, [string]$adapterDir, [string]$promptsJsonl, [string]$outJson) {
  $args = @(
    "--base_model_dir", $baseDir,
    "--prompts_jsonl", $promptsJsonl,
    "--out_json", $outJson,
    "--temperature", "0.0",
    "--max_new_tokens", "240"
  )
  if (-not [string]::IsNullOrWhiteSpace($adapterDir)) {
    $args += @("--adapter_dir", $adapterDir)
  }
  & python .\08_eval_code_assistant.py @args
  if ($LASTEXITCODE -ne 0) {
    throw "Code eval failed: $outJson"
  }
}

function Get-BestCodeEvalReport([string[]]$reportPaths) {
  $best = $null
  foreach ($p in $reportPaths) {
    if (-not (Test-Path $p)) { continue }
    $obj = Get-Content $p -Raw | ConvertFrom-Json
    $score = [double]$obj.overall_score
    if ($null -eq $best -or $score -gt [double]$best.Score) {
      $best = [PSCustomObject]@{
        ReportPath = [string](Resolve-Path $p)
        Score = $score
        PassRate = [double]$obj.pass_rate
        AdapterDir = [string]$obj.adapter_dir
      }
    }
  }
  return $best
}

$BaseDir = Resolve-LocalPath $BaseDir
$CptDir = Resolve-LocalPath $CptDir
$SeedDir = Resolve-LocalPath $SeedDir
$RepairDir = Resolve-LocalPath $RepairDir
$CodeEvalPrompts = Resolve-LocalPath $CodeEvalPrompts
$SyntheticSftPath = Resolve-LocalPath $SyntheticSftPath

Write-Host "3B CODE ASSISTANT MAX PIPELINE"
Write-Host "  ModelId       : $ModelId"
Write-Host "  BaseDir       : $BaseDir"
Write-Host "  CPTDir        : $CptDir"
Write-Host "  SeedDir       : $SeedDir"
Write-Host "  RepairDir     : $RepairDir"
Write-Host "  EvalTopK      : $EvalTopK"
Write-Host ""

if (-not $SkipDownloadBase) {
  & python .\01_download_base.py --model_id "$ModelId" --output_dir "$BaseDir" --dtype float16
  if ($LASTEXITCODE -ne 0) { throw "Base download failed." }
}

if (-not $SkipCpt) {
  & python .\02_train_base.py `
    --model_dir "$BaseDir" `
    --output_dir "$CptDir" `
    --disable_local_data `
    --hf_source "codeparrot/github-code||train|code|1200000" `
    --hf_code_languages "python,typescript" `
    --hf_require_language_tag `
    --repeat_sources `
    --max_steps "$CptSteps" `
    --learning_rate 1.5e-5 `
    --warmup_ratio 0.03 `
    --per_device_batch_size 1 `
    --grad_accum 32 `
    --block_size 256 `
    --gradient_checkpointing `
    --dtype bfloat16 `
    --no-use_fused_optimizer `
    --logging_steps 20 `
    --save_steps 500 `
    --save_total_limit 6 `
    --disable_sample_logging
  if ($LASTEXITCODE -ne 0) { throw "CPT failed." }
}

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
  & python .\tools\generate_code_review_dataset.py --out_jsonl "$SyntheticSftPath" --count "$SyntheticSftCount"
  if ($LASTEXITCODE -ne 0) { throw "Synthetic SFT generation failed." }
}

$baselineReport = Join-Path $CptDir "base_code_assistant_eval_report.json"
Invoke-CodeEval -baseDir "$CptDir" -adapterDir "" -promptsJsonl "$CodeEvalPrompts" -outJson "$baselineReport"

$bestSeed = $null
if (-not $SkipSeed) {
  & python .\04_train_lora.py `
    --model_dir "$CptDir" `
    --output_dir "$SeedDir" `
    --disable_hf_data `
    --validate_data `
    --chat_format tokenizer `
    --code_fence_hygiene normalize `
    --reject_no_markdown_code_examples `
    --fail_on_duplicate_examples `
    --max_duplicate_example_ratio 0.10 `
    --min_loaded_examples 260 `
    --local_jsonl_glob "samples/sft/code_review_seed.jsonl" `
    --local_jsonl_glob "samples/sft/code_assistant_booster.jsonl" `
    --local_jsonl_glob "samples/sft/code_review_synthetic.jsonl" `
    --local_jsonl_glob "samples/sft/repair_math_logic_coding.jsonl" `
    --local_jsonl_glob "samples/sft/system_styles.jsonl" `
    --local_jsonl_glob "samples/sft/chat_alignment_samples.jsonl" `
    --local_jsonl_glob "samples/sft/formatting_code_fences.jsonl" `
    --local_jsonl_glob "samples/sft/format_constraints_strict.jsonl" `
    --local_jsonl_glob "samples/sft/math_reasoning_micro.jsonl" `
    --max_steps "$SeedSteps" `
    --max_length 1024 `
    --per_device_batch_size 1 `
    --grad_accum 16 `
    --learning_rate 3.5e-5 `
    --gradient_checkpointing `
    --dtype float16 `
    --logging_steps 20 `
    --save_steps 100 `
    --save_total_limit 8 `
    --throughput_mode `
    --disable_sample_logging
  if ($LASTEXITCODE -ne 0) { throw "Seed SFT failed." }

  $seedEvalReport = Join-Path $SeedDir "checkpoint_eval_report.json"
  & python .\05_eval_lora_checkpoints.py `
    --base_model_dir "$CptDir" `
    --adapter_dir "$SeedDir" `
    --max_checkpoints 8 `
    --out_json "$seedEvalReport"
  if ($LASTEXITCODE -ne 0) { throw "Seed checkpoint eval failed." }

  $seedTop = Get-TopCheckpointsFromEvalReport -reportPath "$seedEvalReport" -topK $EvalTopK
  $seedCodeReports = @()
  foreach ($ckpt in $seedTop) {
    $name = Split-Path $ckpt -Leaf
    $out = Join-Path $SeedDir ("code_eval_" + $name + ".json")
    Invoke-CodeEval -baseDir "$CptDir" -adapterDir "$ckpt" -promptsJsonl "$CodeEvalPrompts" -outJson "$out"
    $seedCodeReports += $out
  }
  $bestSeed = Get-BestCodeEvalReport -reportPaths $seedCodeReports
  if ($null -ne $bestSeed) {
    Write-Host "Best SEED checkpoint: $($bestSeed.AdapterDir) score=$([math]::Round([double]$bestSeed.Score,4))"
  }
}

$bestRepair = $null
if ((-not $SkipRepair) -and ($null -ne $bestSeed) -and (-not [string]::IsNullOrWhiteSpace([string]$bestSeed.AdapterDir))) {
  & .\run_lora_targeted_repair.ps1 `
    -ModelDir "$CptDir" `
    -SeedAdapter "$($bestSeed.AdapterDir)" `
    -OutDir "$RepairDir" `
    -MaxLength 1024 `
    -BatchSize 1 `
    -GradAccum 16 `
    -MaxSteps "$RepairExtraSteps" `
    -LearningRate 8e-6 `
    -DType float16 `
    -MinLoadedExamples 260 `
    -MaxDuplicateExampleRatio 0.10
  if ($LASTEXITCODE -ne 0) { throw "Repair stage failed." }

  $repairEvalReport = Join-Path $RepairDir "checkpoint_eval_report.json"
  & python .\05_eval_lora_checkpoints.py `
    --base_model_dir "$CptDir" `
    --adapter_dir "$RepairDir" `
    --max_checkpoints 8 `
    --out_json "$repairEvalReport"
  if ($LASTEXITCODE -ne 0) { throw "Repair checkpoint eval failed." }

  $repairTop = Get-TopCheckpointsFromEvalReport -reportPath "$repairEvalReport" -topK $EvalTopK
  $repairCodeReports = @()
  foreach ($ckpt in $repairTop) {
    $name = Split-Path $ckpt -Leaf
    $out = Join-Path $RepairDir ("code_eval_" + $name + ".json")
    Invoke-CodeEval -baseDir "$CptDir" -adapterDir "$ckpt" -promptsJsonl "$CodeEvalPrompts" -outJson "$out"
    $repairCodeReports += $out
  }
  $bestRepair = Get-BestCodeEvalReport -reportPaths $repairCodeReports
  if ($null -ne $bestRepair) {
    Write-Host "Best REPAIR checkpoint: $($bestRepair.AdapterDir) score=$([math]::Round([double]$bestRepair.Score,4))"
  }
}

$seedCompare = Join-Path $SeedDir "compare_vs_baseline.json"
if ($null -ne $bestSeed) {
  & python .\09_compare_code_assistant_reports.py `
    --baseline_report "$baselineReport" `
    --candidate_report "$($bestSeed.ReportPath)" `
    --out_json "$seedCompare"
  if ($LASTEXITCODE -ne 0) { throw "Seed comparison failed." }
}

$repairCompare = Join-Path $RepairDir "compare_vs_baseline.json"
if ($null -ne $bestRepair) {
  & python .\09_compare_code_assistant_reports.py `
    --baseline_report "$baselineReport" `
    --candidate_report "$($bestRepair.ReportPath)" `
    --out_json "$repairCompare"
  if ($LASTEXITCODE -ne 0) { throw "Repair comparison failed." }
}

$recommended = $null
if ($null -ne $bestSeed) {
  $recommended = $bestSeed
}
if (($null -ne $bestRepair) -and (($null -eq $recommended) -or ([double]$bestRepair.Score -gt [double]$recommended.Score))) {
  $recommended = $bestRepair
}

$summary = [ordered]@{
  timestamp = (Get-Date).ToString("s")
  base_dir = $BaseDir
  cpt_dir = $CptDir
  baseline_eval_report = $baselineReport
  best_seed = if ($null -ne $bestSeed) { [ordered]@{ adapter_dir = $bestSeed.AdapterDir; score = $bestSeed.Score; pass_rate = $bestSeed.PassRate; report = $bestSeed.ReportPath } } else { $null }
  best_repair = if ($null -ne $bestRepair) { [ordered]@{ adapter_dir = $bestRepair.AdapterDir; score = $bestRepair.Score; pass_rate = $bestRepair.PassRate; report = $bestRepair.ReportPath } } else { $null }
  recommended = if ($null -ne $recommended) { [ordered]@{ adapter_dir = $recommended.AdapterDir; score = $recommended.Score; pass_rate = $recommended.PassRate; report = $recommended.ReportPath } } else { $null }
  comparisons = [ordered]@{
    seed_vs_baseline = if (Test-Path $seedCompare) { $seedCompare } else { "" }
    repair_vs_baseline = if (Test-Path $repairCompare) { $repairCompare } else { "" }
  }
}

$summaryPath = Join-Path $CptDir "code_assistant_max_summary.json"
$summary | ConvertTo-Json -Depth 6 | Set-Content $summaryPath

Write-Host ""
Write-Host "MAX pipeline complete."
Write-Host "Summary: $summaryPath"
if ($null -ne $recommended) {
  Write-Host "Recommended adapter: $($recommended.AdapterDir)"
  Write-Host "Recommended score  : $([math]::Round([double]$recommended.Score,4))"
}

Pop-Location
