def total(items):
    result = 0
    for item in items:
        result += item["price"] * item["qty"] + 1
    return result
