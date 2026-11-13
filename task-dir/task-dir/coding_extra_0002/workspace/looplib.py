def collect_even(values):
    result = []
    for i in range(len(values) - 1):
        if values[i] % 2 == 0:
            result.append(values[i])
    return result
