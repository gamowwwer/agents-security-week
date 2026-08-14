#include <iostream>
#include <string>

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::string line;
    if (!std::getline(std::cin, line)) return 0;

    // Парсим M (число токенов), игнорируя возможные пробелы/CR.
    long long M = 0;
    {
        bool any = false;
        for (char c : line) {
            if (c >= '0' && c <= '9') { M = M * 10 + (c - '0'); any = true; }
        }
        if (!any) return 0;
    }

    std::string out;
    // Резерв ограничен разумным потолком: даже при некорректно большом M
    // предвыделение не приведёт к попытке захватить гигантский объём памяти.
    long long reserveLines = M < 100000 ? M : 100000;
    out.reserve(static_cast<size_t>(reserveLines) * 8 + 16);

    for (long long i = 0; i < M; ++i) {
        // Если вход исчерпан — больше токенов нет, безопасно останавливаемся.
        // Это гарантирует, что время работы ограничено размером ввода при любом M.
        if (!std::getline(std::cin, line)) break;

        // Убираем возможный завершающий '\r' (Windows-переводы строк).
        if (!line.empty() && line.back() == '\r') line.pop_back();

        bool valid = true;
        int digitCount = 0;
        int sum = 0;
        // Позицию считаем справа. Идём по строке слева направо, поэтому
        // сначала соберём цифры, затем применим Луна справа налево.
        // Чтобы не создавать лишнюю строку, обработаем в два прохода по line,
        // но проще собрать цифры в локальный буфер фиксированного размера.
        // Длина строки <= 100, значит цифр <= 100.
        char digits[128];

        for (char c : line) {
            if (c == '-' || c == ' ') continue;
            if (c >= '0' && c <= '9') {
                digits[digitCount++] = static_cast<char>(c - '0');
            } else {
                valid = false;
                break;
            }
        }

        if (!valid || digitCount == 0) {
            out += "INVALID\n";
            continue;
        }

        // Алгоритм Луна: справа налево, удваиваем каждую вторую (2-я, 4-я, ... справа).
        for (int k = 0; k < digitCount; ++k) {
            int d = digits[digitCount - 1 - k];
            if (k & 1) {                 // k=1 -> 2-я справа, k=3 -> 4-я справа, ...
                d <<= 1;                 // удваиваем
                if (d > 9) d -= 9;       // корректируем
            }
            sum += d;
        }

        out += (sum % 10 == 0) ? "VALID\n" : "INVALID\n";
    }

    std::cout << out;
    return 0;
}
