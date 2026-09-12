# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

# Dot-source this script after activating the Python environment:
#     . ./packaging/activate_cuda.ps1
# NVCC needs MSVC's executable, headers, and libraries in the same process.
[CmdletBinding()]
param()

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw 'Install Visual Studio Build Tools with Desktop development with C++ (vswhere.exe was not found).'
}
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $installation) {
    throw 'No Visual Studio installation with the x64 C++ build tools was found.'
}
$compilerBatch = Join-Path $installation 'VC/Auxiliary/Build/vcvars64.bat'
$compilerEnvironment = & $env:ComSpec /d /c ('call "' + $compilerBatch + '" >nul && set')
if ($LASTEXITCODE -ne 0) {
    throw 'Visual Studio could not initialize its x64 compiler environment.'
}
# Some launchers supply both PATH and Path. vcvars updates PATH, but importing
# the later stale Path entry would overwrite it in PowerShell's environment.
$compilerPath = $null
foreach ($entry in $compilerEnvironment) {
    if ($entry -match '^PATH=(.*)$' -and (-not $compilerPath -or $entry.StartsWith('PATH=', [StringComparison]::Ordinal))) {
        $compilerPath = $matches[1]
    }
    if ($entry -match '^([^=]+)=(.*)$' -and $matches[1] -ine 'PATH') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}
if (-not $compilerPath) {
    throw 'Visual Studio did not return its compiler PATH.'
}
# Remove both case variants before setting one canonical entry. Otherwise a
# child Python process can receive the stale duplicate even when Get-Command
# in this PowerShell session finds cl.exe.
foreach ($key in @([Environment]::GetEnvironmentVariables('Process').Keys)) {
    if ($key -ieq 'PATH') {
        [Environment]::SetEnvironmentVariable($key, $null, 'Process')
    }
}
$env:PATH = $compilerPath
if (-not (Get-Command nvcc.exe -ErrorAction SilentlyContinue) -and $env:CUDA_PATH) {
    $env:PATH = (Join-Path $env:CUDA_PATH 'bin') + ';' + $env:PATH
}
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw 'The MSVC compiler is still unavailable after environment initialization.'
}
if (-not (Get-Command nvcc.exe -ErrorAction SilentlyContinue)) {
    throw 'Install a compatible CUDA Toolkit and put its bin directory on PATH (or set CUDA_PATH).'
}
Write-Host 'CUDA and the MSVC x64 compiler are ready in this PowerShell session.'
