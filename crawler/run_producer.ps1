param(
    [string]$InputPath,
    [string]$BootstrapServers = "localhost:9092",
    [string]$Topic = "raw-comment",
    [string]$LogFile,
    [int]$DeliveryTimeoutMs = 60000
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $InputPath) {
    $InputPath = Join-Path $ProjectRoot "data\raw\performance"
}
if (-not $LogFile) {
    $LogFile = Join-Path $ProjectRoot "logs\producer.log"
}
$PythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Project Python environment is missing at $PythonPath"
}
$ProducerPath = Join-Path $PSScriptRoot "kafka_producer.py"

& $PythonPath -X utf8 $ProducerPath `
    --input $InputPath `
    --bootstrap-servers $BootstrapServers `
    --topic $Topic `
    --log-file $LogFile `
    --delivery-timeout-ms $DeliveryTimeoutMs
exit $LASTEXITCODE
