
a = 10

def int2bin(n):
    bin = []
    while True:
        numerator = n/2
        rem = n%2
        bin.append(rem)
        n = numerator
        if n == 1:
            bin.append(n)
            break
    return bin

print(int2bin(a))
