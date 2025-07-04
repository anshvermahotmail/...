let currentSecret = '';

function generateSecret() {
  const secret = otplib.authenticator.generateSecret();
  currentSecret = secret;

  const otpauth = otplib.authenticator.keyuri(
    'user@example.com', // fake email or username
    'GitHub2FA-Demo',
    secret
  );

  // Generate QR Code via Google Charts
  const qrUrl = `https://chart.googleapis.com/chart?chs=200x200&cht=qr&chl=${encodeURIComponent(otpauth)}`;

  document.getElementById('secret').textContent = secret;
  document.getElementById('qrcode').innerHTML = `<img src="${qrUrl}" alt="Scan QR Code">`;
}

function verifyToken() {
  const token = document.getElementById('token').value;

  const isValid = otplib.authenticator.check(token, currentSecret);

  document.getElementById('result').textContent = isValid
    ? '? Code is valid!'
    : '? Invalid code. Try again.';
}
