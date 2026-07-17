param(
    [string[]] $Configs = @("configs/small_debug.yaml"),
    [int[]] $Seeds = @(42, 123),
    [string] $PythonBin = "C:\Users\Lenovo\.conda\envs\myEnv\python.exe",
    [string] $OutputRoot = "",
    [switch] $SkipWeatherGif
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

foreach ($config in $Configs) {
    $configPath = if ([System.IO.Path]::IsPathRooted($config)) {
        $config
    } else {
        Join-Path $ProjectRoot $config
    }

    foreach ($seed in $Seeds) {
        Write-Host "==> Generating config=$configPath seed=$seed"
        $args = @(
            (Join-Path $ProjectRoot "scripts\generate_static_world.py"),
            "--config", $configPath,
            "--seed", "$seed"
        )
        if ($OutputRoot) {
            $args += @("--output", $OutputRoot)
        }
        if ($SkipWeatherGif) {
            $args += @("--skip-weather-gif")
        }
        & $PythonBin @args
    }
}
