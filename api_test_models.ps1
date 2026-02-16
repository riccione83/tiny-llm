$hostBase = "http://127.0.0.1:8001"
$chatUrl = "$hostBase/v1/chat/completions"

# 1) Prendi tutti i modelli disponibili
$allModels = (Invoke-RestMethod "$hostBase/v1/models").data.id

# 2) Seleziona SOLO quelli che vuoi testare
# $modelsToTest = $allModels | Where-Object {
#     $_ -in @("tiny-llm-0.5b","tiny-llm-3b","tiny-llm-7b","base-qwen-0.5b","base-qwen-3b")
# }

$modelsToTest = $allModels | Where-Object {
    $_ -in @("tiny-llm-7b")
}


# 3) Funzione per chiamare il modello
function Ask($model, $content, $temp=0.2, $max=200) {
    $body = @{
        model = $model
        messages = @(@{ role="user"; content=$content })
        temperature = $temp
        max_tokens = $max
    } | ConvertTo-Json -Depth 6

    try {
        $resp = Invoke-RestMethod -Uri $chatUrl -Method Post -ContentType "application/json" -Body $body -ErrorAction Stop
        if ($null -ne $resp.choices -and $resp.choices.Count -gt 0) {
            return @{
                ok = $true
                content = $resp.choices[0].message.content
                model = $resp.model
                latency_ms = $resp.latency_ms
                tokens = $resp.tokens
            }
        }
        return @{
            ok = $false
            error = "Missing choices in response"
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

# 4) Test suite
$tests = @(
    @{
        name = "JSON strict"
        prompt = 'Return ONLY valid JSON {"sum": number}. Task: sum 4 and 5'
        temp = 0
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
where is the issue here?

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
        name = "hard reasoning"
        prompt = @"
Return ONLY valid JSON with keys: language (string), has_code (boolean), code (string). Task: write add(a,b) in Python. No markdown.
"@
        temp = 0.1
        max = 200
    }

)

# 5) Loop su modelli e test
foreach ($m in $modelsToTest) {

    Write-Host "`n==============================="
    Write-Host "MODEL: $m"
    Write-Host "==============================="

    foreach ($t in $tests) {
        Write-Host "`n--- $($t.name) ---"
        $result = Ask $m $t.prompt $t.temp $t.max
        if ($result.ok) {
            Write-Host ("[OK] model={0} latency_ms={1} tokens={2}" -f $result.model, $result.latency_ms, $result.tokens)
            Write-Host $result.content
        } else {
            Write-Host ("[ERROR] {0}" -f $result.error)
        }
    }
}
