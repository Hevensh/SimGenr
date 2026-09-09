param(
    [string]$Python = 'D:\anaconda\python.exe',
    [switch]$GeneratorOnly
)

# Process-local settings only. Neither Conda nor global environment is changed.
$repoRoot = Split-Path $PSScriptRoot -Parent
$previousLocation = Get-Location
$variableNames = @('PYTHONDONTWRITEBYTECODE','PYTHONPATH','MPLCONFIGDIR','TEMP','TMP',
                   'MKL_THREADING_LAYER','MKL_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')
$oldValues = @{}
foreach ($name in $variableNames) { $oldValues[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
try {
    Set-Location -LiteralPath $repoRoot
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:MPLCONFIGDIR = Join-Path $repoRoot 'outputs/.mplconfig'
    $env:TEMP = Join-Path $repoRoot 'outputs/.tmp'
    $env:TMP = $env:TEMP
    $env:MKL_THREADING_LAYER = 'SEQUENTIAL'
    $env:MKL_NUM_THREADS = '1'
    $env:OMP_NUM_THREADS = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
    New-Item -ItemType Directory -Force $env:TEMP | Out-Null
    $localDependencies = Join-Path $repoRoot 'outputs/.test-deps'
    if (Test-Path -LiteralPath $localDependencies) {
        $env:PYTHONPATH = $localDependencies + [IO.Path]::PathSeparator + $repoRoot
    }
    $testArguments = @('-B','-m','pytest','-q','-p','no:cacheprovider','tests',
                       '--basetemp=outputs/.pytest_tmp','--junitxml=outputs/physics_tests.xml')
    if ($GeneratorOnly) { $testArguments += '--ignore=tests/test_forecasting_model.py' }
    & $Python @testArguments
    $result = $LASTEXITCODE
}
finally {
    foreach ($name in $variableNames) { [Environment]::SetEnvironmentVariable($name, $oldValues[$name], 'Process') }
    Set-Location -LiteralPath $previousLocation
}
exit $result
