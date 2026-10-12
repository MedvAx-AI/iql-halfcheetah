param(
    [Parameter(Mandatory = $true)][string]$RecordingPath,
    [ValidatePattern('^[A-Za-z0-9_-]+$')][string]$GroupName = 'IQL_HalfCheetah',
    [ValidateSet('pptx', 'pdf')][string]$PresentationFormat = 'pptx',
    [string]$OutputDirectory = $PSScriptRoot
)
$ErrorActionPreference = 'Stop'
$video = Get-Item -LiteralPath $RecordingPath
if ($video.PSIsContainer -or $video.Length -eq 0 -or $video.Extension.ToLowerInvariant() -notin @('.mp4', '.mov', '.mkv', '.webm', '.avi')) {
    throw 'Choose your non-empty narrated presentation recording (.mp4 recommended).'
}
$deck = Get-Item -LiteralPath (Join-Path $PSScriptRoot "IQL_RL_Proposal.$PresentationFormat")
if (-not (Test-Path -LiteralPath $OutputDirectory -PathType Container)) {
    New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
}
$destination = Join-Path (Resolve-Path -LiteralPath $OutputDirectory).Path "${GroupName}_RL_Proposal.zip"
if (Test-Path -LiteralPath $destination) { throw "ZIP already exists: $destination. Choose a different group name or output folder." }
Compress-Archive -LiteralPath @($deck.FullName, $video.FullName) -DestinationPath $destination
Write-Output "Created: $destination"
Write-Output 'Before Moodle upload: open the ZIP and verify the full narrated video plays, lasts 5-7 minutes, and shows the slides.'
