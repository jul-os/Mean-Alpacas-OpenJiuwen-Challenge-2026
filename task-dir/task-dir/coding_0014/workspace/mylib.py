def word_count(text):
    words = text.lower().split(" ")
    out = {}
    for w in words:
        out[w] = out.get(w, 0) + 1
    return out
