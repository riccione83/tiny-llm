param(
    [ValidateSet("hf","tiny")]
    [string]$Backend = "hf",
    [string]$ModelName = "Qwen/Qwen2.5-1.5B-Instruct",
    [string]$EmbeddingModel = "sentence-transformers/all-MiniLM-L6-v2",
    [string]$TinyCkpt = "checkpoints_v2/final.pt",
    [string]$TinyTokenizer = "tokenizer.model",
    [string]$TinyLora = "",
    [double]$TinyTopP = 1.0,
    [double]$Temperature = 0.0,
    [int]$TopK = 5,
    [int]$SearchResults = 5
)

$ErrorActionPreference = "Stop"

Write-Host "Installing dependencies..."
python -m pip install -U pip
python -m pip install -r requirements.txt

if ($Backend -eq "hf") {
    Write-Host "Pre-downloading LLM..."
    python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; m='$ModelName'; AutoTokenizer.from_pretrained(m); AutoModelForCausalLM.from_pretrained(m)"
}

Write-Host "Pre-downloading embedding model..."
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('$EmbeddingModel')"

Write-Host "Starting mini assistant..."
if ($Backend -eq "tiny") {
    python -m mini_assistant.chat --backend tiny --tiny_ckpt $TinyCkpt --tiny_tokenizer $TinyTokenizer --tiny_lora $TinyLora --tiny_top_p $TinyTopP --embedding_model $EmbeddingModel --temperature $Temperature --top_k $TopK --search_results $SearchResults
} else {
    python -m mini_assistant.chat --backend hf --model_name $ModelName --embedding_model $EmbeddingModel --temperature $Temperature --top_k $TopK --search_results $SearchResults
}
