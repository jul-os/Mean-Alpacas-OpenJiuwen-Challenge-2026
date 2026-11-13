from pricing import discounted

def total(items, discount):
    return sum(discounted(price, discount) * quantity for price, quantity in items)
