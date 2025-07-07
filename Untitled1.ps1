# Function to map the numeric attributes to readable names
function Get-HumanReadableAttributes {
    param (
        [Parameter(Mandatory=$true)]
        [System.IO.FileInfo]$File
    )
    
    $attributes = $File.Attributes
    $attributeFlags = @()

    # Check each attribute flag and map to readable names
    if ($attributes -band [System.IO.FileAttributes]::ReadOnly) { $attributeFlags += "ReadOnly" }
    if ($attributes -band [System.IO.FileAttributes]::Hidden) { $attributeFlags += "Hidden" }
    if ($attributes -band [System.IO.FileAttributes]::System) { $attributeFlags += "System" }
    if ($attributes -band [System.IO.FileAttributes]::Directory) { $attributeFlags += "Directory" }
    if ($attributes -band [System.IO.FileAttributes]::Archive) { $attributeFlags += "Archive" }
    if ($attributes -band [System.IO.FileAttributes]::Device) { $attributeFlags += "Device" }
    if ($attributes -band [System.IO.FileAttributes]::Normal) { $attributeFlags += "Normal" }
    if ($attributes -band [System.IO.FileAttributes]::Temporary) { $attributeFlags += "Temporary" }
    if ($attributes -band [System.IO.FileAttributes]::SparseFile) { $attributeFlags += "SparseFile" }
    if ($attributes -band [System.IO.FileAttributes]::ReparsePoint) { $attributeFlags += "ReparsePoint" }
    if ($attributes -band [System.IO.FileAttributes]::Compressed) { $attributeFlags += "Compressed" }
    if ($attributes -band [System.IO.FileAttributes]::Offline) { $attributeFlags += "Offline" }
    if ($attributes -band [System.IO.FileAttributes]::NotContentIndexed) { $attributeFlags += "NotContentIndexed" }
    if ($attributes -band [System.IO.FileAttributes]::Encrypted) { $attributeFlags += "Encrypted" }
    if ($attributes -band [System.IO.FileAttributes]::IntegrityStream) { $attributeFlags += "IntegrityStream" }
    if ($attributes -band [System.IO.FileAttributes]::Virtual) { $attributeFlags += "Virtual" }
    if ($attributes -band [System.IO.FileAttributes]::NoScrubData) { $attributeFlags += "NoScrubData" }

    # Return the human-readable attribute flags
    return $attributeFlags -join ", "
}

# Function to check file's download status (whether it's fully downloaded or not)
function Get-DownloadStatus {
    param (
        [Parameter(Mandatory=$true)]
        [System.IO.FileInfo]$File
    )
    
    # Get the file size
    $fileSize = $File.Length

    # Check if the file has the Offline attribute (cloud-only file)
    $isOffline = $File.Attributes -band [System.IO.FileAttributes]::Offline

    # If the file is not Offline, it has been downloaded
    if ($isOffline) {
        return "File is not fully downloaded yet"
    } else {
        return "File is fully downloaded"
    }
}

# Main command to filter and display .ova files with their attributes and download status
Get-ChildItem -Path C:\ -Filter *.ova -Recurse -File -ErrorAction SilentlyContinue |
Where-Object { ($_.Attributes.ToString() -notmatch "Offline") } |
Select-Object FullName, Length, @{Name="Attributes";Expression={ Get-HumanReadableAttributes $_ }},
                       @{Name="Download Status";Expression={ Get-DownloadStatus $_ }}
