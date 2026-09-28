# Starts both mock APIs in their own windows, with labenv active
$root = $PSScriptRoot
foreach ($shim in "snow_shim", "jira_shim") {
    Start-Process powershell -WorkingDirectory $root -ArgumentList "-NoExit", "-Command",
        "& '$root\labenv\Scripts\Activate.ps1'; python mcp_server\$shim.py"
}
