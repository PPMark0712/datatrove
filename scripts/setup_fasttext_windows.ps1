param(
    [string]$Python = 'python',
    [string]$ToolchainSetup = 'C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $ToolchainSetup)) {
    throw 'Install Microsoft C++ Build Tools before running this script.'
}
$taskPython = (Get-Command $Python -ErrorAction Stop).Source
& $taskPython -m pip install --index-url https://pypi.org/simple 'pybind11>=2.12,<3'
if ($LASTEXITCODE -ne 0) { throw 'pybind11 installation failed' }

$taskCache = Join-Path (Split-Path $PSScriptRoot -Parent) 'webui\.cache'
New-Item -ItemType Directory -Path $taskCache -Force | Out-Null
$taskHeader = Join-Path $taskCache 'fasttext_win_compat.h'
@'
#include <cstddef>
using ssize_t = std::ptrdiff_t;
'@ | Set-Content -LiteralPath $taskHeader -Encoding ascii

$taskOldSDK = $env:DISTUTILS_USE_SDK
$taskOldMSSDK = $env:MSSdk
$taskOldCL = $env:CL
try {
    $env:DISTUTILS_USE_SDK = '1'
    $env:MSSdk = '1'
    $env:CL = '/std:c++17 /utf-8 /FI"{0}"' -f $taskHeader
    # The upstream extension uses POSIX ssize_t without a Windows definition.
    # Supply a pointer-sized alias without changing its Python/NumPy2 code.
    $taskCommand = 'call "{0}" && "{1}" -m pip install --index-url https://pypi.org/simple --no-build-isolation fasttext-numpy2-wheel' -f $ToolchainSetup, $taskPython
    & cmd.exe /d /s /c $taskCommand
    if ($LASTEXITCODE -ne 0) { throw 'fastText build failed' }
} finally {
    $env:DISTUTILS_USE_SDK = $taskOldSDK
    $env:MSSdk = $taskOldMSSDK
    $env:CL = $taskOldCL
}
& $taskPython -c "from datatrove.pipeline.filters import LanguageFilter; LanguageFilter(languages=['en']); print('fastText dependency check passed')"
if ($LASTEXITCODE -ne 0) { throw 'fastText dependency check failed' }
