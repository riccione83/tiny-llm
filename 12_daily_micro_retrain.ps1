param(
    [string]$BaseCkpt = "checkpoints_v2/final.pt",
    [string]$Tokenizer = "tokenizer.model",
    [string]$OutDir = "finetuning_v2",
    [string]$TurnLog = "data/chat_turns_log.jsonl",
    [string]$WebLog = "data/web_chat_log.jsonl",
    [string]$OutNew = "data/daily_new_sft.jsonl",
    [string]$OutSft = "data/sft_daily_micro.jsonl",
    [int]$MaxNewRows = 30000,
    [int]$MaxRows = 120000,
    [double]$NewRatio = 0.25,
    [double]$SummarizeRatio = 0.45,
    [double]$RoutingRatio = 0.15,
    [int]$Seed = 42,
    [double]$Lr = 3e-5,
    [int]$Warmup = 50,
    [int]$MaxOptSteps = 1200,
    [switch]$NoTrain
)

$ErrorActionPreference = "Stop"

python 11_make_daily_micro_sft.py `
  --turn_log $TurnLog `
  --web_log $WebLog `
  --max_new_rows $MaxNewRows `
  --out_new $OutNew `
  --out $OutSft `
  --max_rows $MaxRows `
  --new_ratio $NewRatio `
  --summarize_ratio $SummarizeRatio `
  --routing_ratio $RoutingRatio `
  --seed $Seed

if ($NoTrain) {
    Write-Host "Dataset built only (NoTrain)."
    exit 0
}

python 08_train_summarize_lora.py `
  --base_ckpt $BaseCkpt `
  --tokenizer $Tokenizer `
  --sft_jsonl $OutSft `
  --out_dir $OutDir `
  --epochs 1 `
  --batch_size 16 `
  --grad_accum 8 `
  --lr $Lr `
  --warmup $Warmup `
  --print_every 20 `
  --sample_every 100 `
  --save_every 200 `
  --resume `
  --resume_model_only `
  --max_opt_steps $MaxOptSteps

