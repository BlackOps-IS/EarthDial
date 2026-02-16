# Deploy script for EarthDial to VPS
$password = "MAUsqZ80"
$commands = @"
cd /opt/EarthDial
git pull
systemctl restart earthdial
sleep 2
systemctl status earthdial --no-pager
"@

# Using plink if available, otherwise manual SSH
if (Get-Command plink -ErrorAction SilentlyContinue) {
    echo $password | plink -batch -pw $password root@216.250.115.87 $commands
} else {
    Write-Host "==> Manual deployment required. Run these commands on VPS:"
    Write-Host ""
    Write-Host "ssh root@216.250.115.87"
    Write-Host "Password: $password"
    Write-Host ""
    Write-Host $commands
}
