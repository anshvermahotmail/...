// Base32 encoding/decoding helper
const base32chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";

function base32tohex(base32) {
  let bits = "";
  let hex = "";

  for (let i = 0; i < base32.length; i++) {
    const val = base32chars.indexOf(base32.charAt(i).toUpperCase());
    bits += val.toString(2).padStart(5, "0");
  }

  for (let i = 0; i + 4 <= bits.length; i += 4) {
    const chunk = bits.substr(i, 4);
    hex += parseInt(chunk, 2).toString(16);
  }
  return hex;
}

function leftpad(str, len, pad) {
  return (new Array(len + 1).join(pad) + str).slice(-len);
}

// Generate TOTP code using jsSHA HMAC-SHA1
function generateTOTP(secret) {
  const epoch = Math.floor(Date.now() / 1000);
  const time = leftpad(Math.floor(epoch / 30).toString(16), 16, '0');
  const key = base32tohex(secret);

  const shaObj = new jsSHA("SHA-1", "HEX");
  shaObj.setHMACKey(key, "HEX");
  shaObj.update(time);
  const hmac = shaObj.getHMAC("HEX");

  const offset = parseInt(hmac.substring(hmac.length - 1), 16);

  const part = hmac.substr(offset * 2, 8);
  let code = (parseInt(part, 16) & 0x7fffffff) + "";
  code = code.substr(code.length - 6, 6);

  return leftpad(code, 6, '0');
}

let currentSecret = "";

function generateSecret() {
  currentSecret = "";
  for (let i = 0; i < 16; i++) {
    currentSecret += base32chars.charAt(Math.floor(Math.random() * base32chars.length));
  }

  document.getElementById("secret").textContent = currentSecret;

  const otpauth = `otpauth://totp/Example:user@example.com?secret=${currentSecret}&issuer=Example`;

  QRCode.toCanvas(document.getElementById("qrcode"), otpauth, function (error) {
    if (error) console.error(error);
  });
}

function verifyToken() {
  const token = document.getElementById("token").value.trim();

  if (token === generateTOTP(currentSecret)) {
    document.getElementById("result").textContent = "✅ Code is valid!";
    document.getElementById("result").style.color = "green";
  } else {
    document.getElementById("result").textContent = "❌ Invalid code. Try again.";
    document.getElementById("result").style.color = "red";
  }
}
