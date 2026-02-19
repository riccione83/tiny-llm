param(
    [string]$BaseUrl = "http://127.0.0.1:8001",
    [string[]]$Models = @("tiny-llm-7b")
)

$ErrorActionPreference = "Stop"

$chatUrl = "$BaseUrl/v1/chat/completions"
$modelsUrl = "$BaseUrl/v1/models"

function Invoke-ModelChat([string]$Model, [string]$Prompt, [double]$Temperature = 0.2, [int]$MaxTokens = 200) {
    $body = @{
        model = $Model
        messages = @(@{ role = "user"; content = $Prompt })
        temperature = $Temperature
        max_tokens = $MaxTokens
    } | ConvertTo-Json -Depth 6

    try {
        $resp = Invoke-RestMethod -Uri $chatUrl -Method Post -ContentType "application/json" -Body $body
        if ($null -ne $resp.choices -and $resp.choices.Count -gt 0) {
            return @{
                ok = $true
                model = $resp.model
                latency_ms = $resp.latency_ms
                tokens = $resp.tokens
                content = $resp.choices[0].message.content
            }
        }
        return @{
            ok = $false
            error = "Missing choices in API response."
            raw = $resp
        }
    } catch {
        $msg = $_.Exception.Message
        if ($_.ErrorDetails -and $_.ErrorDetails.Message) {
            $msg = $_.ErrorDetails.Message
        }
        return @{
            ok = $false
            error = $msg
        }
    }
}

$available = @((Invoke-RestMethod $modelsUrl).data.id)
$selected = @($available | Where-Object { $_ -in $Models })

if ($selected.Count -eq 0) {
    Write-Host "[ERROR] None of the requested models are exposed by $BaseUrl"
    Write-Host "Requested: $($Models -join ', ')"
    Write-Host "Available: $($available -join ', ')"
    exit 1
}

$tests = @(
    @{
        name = "JSON strict"
        prompt = 'Return ONLY valid JSON {"sum": number}. Task: sum 4 and 5'
        temp = 0.0
        max = 80
    },
    @{
        name = "Code fix"
        prompt = "Fix this code: def add(a,b): return a-b"
        temp = 0.1
        max = 200
    },
    @{
        name = "Bug reasoning"
        prompt = @"
Where is the issue here?

def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 5:
            return False
    return True
"@
        temp = 0.1
        max = 200
    },
    @{
        name = "Structured output"
        prompt = 'Return ONLY valid JSON with keys: language (string), has_code (boolean), code (string). Task: write add(a,b) in Python. No markdown.'
        temp = 0.1
        max = 200
    }
)

foreach ($model in $selected) {
    Write-Host "`n==============================="
    Write-Host "MODEL: $model"
    Write-Host "==============================="

    foreach ($test in $tests) {
        Write-Host "`n--- $($test.name) ---"
        $result = Invoke-ModelChat -Model $model -Prompt $test.prompt -Temperature $test.temp -MaxTokens $test.max
        if ($result.ok) {
            Write-Host ("[OK] model={0} latency_ms={1} tokens={2}" -f $result.model, $result.latency_ms, $result.tokens)
            Write-Host $result.content
        } else {
            Write-Host ("[ERROR] {0}" -f $result.error)
        }
    }
}
