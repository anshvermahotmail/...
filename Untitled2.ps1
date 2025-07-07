# Function to get human-readable attributes
function Get-HumanReadableAttributes {
    param (
        [System.IO.FileInfo]$File
    )
    $flags = @()
    $attr = $File.Attributes

    foreach ($name in [Enum]::GetNames([System.IO.FileAttributes])) {
        if ($attr.HasFlag([System.IO.FileAttributes]::$name)) {
            $flags += $name
        }
    }
    return $flags -join ", "
}

# Function to get Size on Disk (using FSUTIL if available, fallback to length)
function Get-SizeOnDisk {
    param (
        [string]$FilePath
    )

    try {
        $output = fsutil file queryfileid "$FilePath" 2>$null
        if ($?) {
            $blockSize = (fsutil fsinfo ntfsinfo (Get-Item $FilePath).Directory.Root.Name).Split("`n") |
                         Where-Object { $_ -match "Bytes Per Cluster" } |
                         ForEach-Object { ($_ -split ":")[1].Trim() } |
                         Select-Object -First 1

            $length = (Get-Item $FilePath).Length
            $blockSize = [int]$blockSize
            $blocks = [math]::Ceiling($length / $blockSize)
            return $blocks * $blockSize
        } else {
            return (Get-Item $FilePath).Length
        }
    } catch {
        return (Get-Item $FilePath).Length
    }
}

# Main logic
Get-ChildItem -Path C:\ -Filter *.ova -Recurse -File -ErrorAction SilentlyContinue |
ForEach-Object {
    $file = $_
    $size = $file.Length
    $sizeOnDisk = Get-SizeOnDisk -FilePath $file.FullName
    $attributes = Get-HumanReadableAttributes -File $file
    $isOffline = $file.Attributes -band [System.IO.FileAttributes]::Offline
    $progress = if ($size -gt 0) { [math]::Round(($sizeOnDisk / $size) * 100, 2) } else { 0 }

    [PSCustomObject]@{
        FullName         = $file.FullName
        Size_Bytes       = $size
        SizeOnDisk_Bytes = $sizeOnDisk
        Attributes       = $attributes
        IsOffline        = [bool]$isOffline
        DownloadProgress = "$progress %"
    }
} | Format-Table -AutoSize
