import sys
from math import gcd
from functools import reduce

def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    n = int(data[0])
    nums = list(map(int, data[1:1+n]))
    P = reduce(gcd, nums)
    sys.stdout.write(str(P) + "\n")

if __name__ == "__main__":
    main()
