param(
    [switch]$VerboseOutput
)

$ErrorActionPreference = "Stop"

if ($VerboseOutput) {
    Write-Host "[info] Running tiny-llm unit tests (verbose mode)..."
    python -m unittest discover -v -s tiny-llm/tests -p "test_*.py"
} else {
    Write-Host "[info] Running tiny-llm unit tests..."
    python -m unittest discover -s tiny-llm/tests -p "test_*.py"
}

Write-Host "[ok] Checks completed."
