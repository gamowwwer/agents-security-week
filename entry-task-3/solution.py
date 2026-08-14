import sys


def main():
    # Read all input as raw bytes so the handshake string H (which may contain
    # spaces and must be treated verbatim) is preserved exactly.
    data = sys.stdin.buffer.read()

    # If stdin is empty, fall back to input.txt
    if not data:
        try:
            with open("input.txt", "rb") as f:
                data = f.read()
        except FileNotFoundError:
            return

    # Split into lines, handling both \n and \r\n. We must NOT strip spaces,
    # only line terminators.
    lines = data.split(b"\n")
    # Remove a trailing \r from each line if present (Windows line endings)
    lines = [ln[:-1] if ln.endswith(b"\r") else ln for ln in lines]

    idx = 0
    # First line: K and M
    first = lines[idx].split()
    idx += 1
    K = int(first[0])
    M = int(first[1])

    # Second line: handshake H, verbatim (kept as raw bytes)
    H = lines[idx]
    idx += 1

    ciphertexts = []
    for _ in range(M):
        hex_str = lines[idx].strip()
        idx += 1
        ct = bytes.fromhex(hex_str.decode("ascii"))
        ciphertexts.append(ct)

    # Recover the key from the first ciphertext using the known plaintext:
    # cipher[j] = plain[j] XOR key[j]  =>  key[j] = cipher[j] XOR H[j]
    first_ct = ciphertexts[0]
    key = bytes(first_ct[j] ^ H[j] for j in range(K))

    # Decrypt each ciphertext. To stay well within the time limit we build a
    # keystream (the repeating key extended to the message length) and XOR the
    # whole message at once using big-integer arithmetic, which runs in C.
    out_lines = []
    for ct in ciphertexts:
        L = len(ct)
        # Keystream aligned to index 0: key repeated, then truncated.
        keystream = (key * (L // K + 1))[:L]
        plain_int = int.from_bytes(ct, "big") ^ int.from_bytes(keystream, "big")
        plain = plain_int.to_bytes(L, "big")
        out_lines.append(plain.decode("ascii"))

    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
