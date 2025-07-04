import base64
import time
import hmac
import hashlib
import subprocess
import shutil
import json
import os

# === Python TOTP Implementation ===
def base32_to_bytes(secret_base32):
    missing_padding = len(secret_base32) % 8
    if missing_padding:
        secret_base32 += '=' * (8 - missing_padding)
    return base64.b32decode(secret_base32, casefold=True)

def generate_totp_debug(secret_base32, timestep=30, digits=6, for_time=None):
    if for_time is None:
        for_time = int(time.time())

    counter = int(for_time // timestep)
    key = base32_to_bytes(secret_base32)
    counter_bytes = counter.to_bytes(8, byteorder='big')
    hmac_hash = hmac.new(key, counter_bytes, hashlib.sha1).digest()
    offset = hmac_hash[-1] & 0x0F
    truncated_hash = hmac_hash[offset:offset + 4]
    code_int = int.from_bytes(truncated_hash, byteorder='big') & 0x7FFFFFFF
    code = str(code_int % (10 ** digits)).zfill(digits)

    return {
        "timestamp": for_time,
        "counter": counter,
        "key_bytes": key.hex(),
        "counter_bytes": counter_bytes.hex(),
        "hmac_hash": hmac_hash.hex(),
        "offset": offset,
        "truncated_hex": truncated_hash.hex(),
        "truncated_int": code_int,
        "totp_code": code
    }

# === Setup ===
secret = "I3Z37WN7TADMOH5F"
ps_cmd = shutil.which("powershell") or shutil.which("pwsh")
if not ps_cmd:
    print("? PowerShell is not available on this system.")
    exit()

def compare_and_print(py_data, ps_data):
    def compare_fields(field):
        py_val = py_data.get(field)
        ps_val = ps_data.get(field)
        status = "?" if str(py_val) == str(ps_val) else "?"
        print(f"{status} {field:<15}: Python = {py_val} | PowerShell = {ps_val}")

    print("\n?? TOTP Comparison Report\n" + "=" * 40)
    for field in [
        "timestamp", "counter", "key_bytes", "counter_bytes",
        "hmac_hash", "offset", "truncated_int", "totp_code"
    ]:
        compare_fields(field)

# === Loop to compare every second ===
try:
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        timestamp = int(time.time())

        # Generate TOTP in Python
        py_data = generate_totp_debug(secret, for_time=timestamp)

        # Generate matching PowerShell script
        powershell_script = f"""
        function Base32ToBytes {{
            param([string]$base32)
            $alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
            $base32 = $base32.ToUpper().Replace('=', '')
            $bits = ""
            foreach ($char in $base32.ToCharArray()) {{
                $val = $alphabet.IndexOf($char)
                $bits += [Convert]::ToString($val, 2).PadLeft(5, '0')
            }}
            $bytes = New-Object System.Collections.Generic.List[Byte]
            for ($i = 0; $i + 8 -le $bits.Length; $i += 8) {{
                $chunk = $bits.Substring($i, 8)
                $bytes.Add([Convert]::ToByte($chunk, 2))
            }}
            return $bytes.ToArray()
        }}

        function Get-TOTP {{
            param (
                [string]$secret = "{secret}",
                [int]$timestamp = {timestamp},
                [int]$digits = 6,
                [int]$step = 30
            )

            $counter = [math]::Floor($timestamp / $step)
            $counterBytes = New-Object Byte[] 8
            $tempCounter = $counter
            for ($i = 7; $i -ge 0; $i--) {{
                $counterBytes[$i] = $tempCounter -band 0xFF
                $tempCounter = $tempCounter -shr 8
            }}

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

            [PSCustomObject]@{{
                timestamp      = $timestamp
                counter        = $counter
                key_bytes      = [BitConverter]::ToString($keyBytes).Replace("-", "").ToLower()
                counter_bytes  = [BitConverter]::ToString($counterBytes).Replace("-", "").ToLower()
                hmac_hash      = [BitConverter]::ToString($hash).Replace("-", "").ToLower()
                offset         = $offset
                truncated_int  = $truncated
                totp_code      = $otp.ToString().PadLeft($digits, '0')
            }} | ConvertTo-Json -Compress
        }}

        Get-TOTP
        """

        ps_result = subprocess.run(
            [ps_cmd, "-Command", powershell_script],
            capture_output=True, text=True
        )

        try:
            ps_data = json.loads(ps_result.stdout.strip())
            compare_and_print(py_data, ps_data)
        except Exception as e:
            print("? PowerShell output parsing failed.")
            print(ps_result.stdout)
            print("Error:", str(e))

        time.sleep(1)

except KeyboardInterrupt:
    print("\n? Exiting...")
