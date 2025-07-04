import time
import hmac
import hashlib
import base64

def base32_to_bytes(secret_base32):
    # Correct padding for Base32
    missing_padding = len(secret_base32) % 8
    if missing_padding:
        secret_base32 += '=' * (8 - missing_padding)
    return base64.b32decode(secret_base32, casefold=True)

def generate_totp(secret_base32, timestep=30, digits=6, for_time=None):
    if for_time is None:
        for_time = int(time.time())
    
    counter = int(for_time // timestep)
    key = base32_to_bytes(secret_base32)

    # 8-byte big-endian counter
    counter_bytes = counter.to_bytes(8, byteorder='big')

    # HMAC-SHA1
    hmac_hash = hmac.new(key, counter_bytes, hashlib.sha1).digest()

    offset = hmac_hash[-1] & 0x0F
    truncated_hash = hmac_hash[offset:offset + 4]

    code_int = int.from_bytes(truncated_hash, byteorder='big') & 0x7FFFFFFF
    code = str(code_int % (10 ** digits)).zfill(digits)

    return code, for_time

# Test
secret = "I3Z37WN7TADMOH5F"
while True:
    code, ts = generate_totp(secret)
    print(f"TOTP Code : {code}")
    print(f"Unix Time : {ts}")
    print(f"Time left : {30 - (ts % 30)}s")
    time.sleep(1)
