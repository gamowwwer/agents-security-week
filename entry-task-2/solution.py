import sys


def main() -> None:
    data = sys.stdin.buffer.read().split(b"\n")
    if not data:
        return

    try:
        m = int(data[0])
    except (ValueError, IndexError):
        return

    out = []
    delete_chars = b"- \r\t"

    for i in range(1, m + 1):
        line = data[i] if i < len(data) else b""
        # Разделители '-' и пробелы (плюс возможные \r/\t) не несут смысла — удаляем
        cleaned = line.translate(None, delete_chars)

        # Пусто или есть нецифровой символ -> INVALID
        if not cleaned or not cleaned.isdigit():
            out.append("INVALID")
            continue

        # Алгоритм Луна: идём справа налево, удваиваем каждую вторую цифру
        total = 0
        double = False
        for j in range(len(cleaned) - 1, -1, -1):
            d = cleaned[j] - 48  # ASCII '0' == 48
            if double:
                d <<= 1
                if d > 9:
                    d -= 9
            total += d
            double = not double

        out.append("VALID" if total % 10 == 0 else "INVALID")

    sys.stdout.write("\n".join(out))
    if out:
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
