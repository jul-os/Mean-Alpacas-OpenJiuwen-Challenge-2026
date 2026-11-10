def ways(n):
    if n < 2:
        return 1
    return ways(n - 1) + ways(n - 2)
