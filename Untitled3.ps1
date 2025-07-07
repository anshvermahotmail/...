function Base32ToBytes {
    param([string]$base32)
    $alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
    $base32 = $base32.ToUpper().Replace('=', '')
    $bits = ""
    foreach ($char in $base32.ToCharArray()) {
        $val = $alphabet.IndexOf($char)
        if ($val -lt 0) {
            throw "Invalid base32 character: $char"
        }
        $bits += [Convert]::ToString($val, 2).PadLeft(5, '0')
    }
    $bytes = New-Object System.Collections.Generic.List[Byte]
    for ($i = 0; $i + 8 -le $bits.Length; $i += 8) {
        $chunk = $bits.Substring($i, 8)
        $bytes.Add([Convert]::ToByte($chunk, 2))
    }
    return $bytes.ToArray()
}

function Get-TOTP {
    param (
        [string]$secret = "I3Z37WN7TADMOH5F",
        [int]$digits = 6,
        [int]$step = 30
    )

    while ($true) {
        Clear-Host

        $timestamp = [int](Get-Date -UFormat %s)
        $counter = [math]::Floor($timestamp / $step)

        $counterBytes = New-Object Byte[] 8
        $tempCounter = $counter
        for ($i = 7; $i -ge 0; $i--) {
            $counterBytes[$i] = $tempCounter -band 0xFF
            $tempCounter = $tempCounter -shr 8
        }

        $keyBytes = Base32ToBytes $secret

        $hmac = New-Object System.Security.Cryptography.HMACSHA1
        $hmac.Key = $keyBytes
        $hash = $hmac.ComputeHash($counterBytes)

        $offset = $hash[$hash.Length - 1] -band 0x0F
        $truncated = (($hash[$offset] -band 0x7F) -shl 24) `
                   -bor (($hash[$offset + 1] -band 0xFF) -shl 16) `
                   -bor (($hash[$offset + 2] -band 0xFF) -shl 8) `
                   -bor ($hash[$offset + 3] -band 0xFF)

        $otp = $truncated % ([math]::Pow(10, $digits))
        $code = $otp.ToString().PadLeft($digits, '0')

        $timeLeft = $step - ($timestamp % $step)

        Write-Host "🔐 Current TOTP Code      : $code"
        Write-Host "⏱️  Seconds Left in Window: $timeLeft"
        Write-Host "📅 Timestamp              : $timestamp"
        Write-Host "🔁 Counter                : $counter"
        Write-Host "🧬 Key Bytes              : $([BitConverter]::ToString($keyBytes).Replace('-', '').ToLower())"
        Write-Host "📦 Counter Bytes          : $([BitConverter]::ToString($counterBytes).Replace('-', '').ToLower())"
        Write-Host "🔒 HMAC SHA1 Hash         : $([BitConverter]::ToString($hash).Replace('-', '').ToLower())"
        Write-Host "📍 Offset                 : $offset"
        Write-Host "🧮 Truncated Integer      : $truncated"

        Start-Sleep -Seconds 1
    }
}

# Run it
Get-TOTP
