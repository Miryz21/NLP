"""Пункт 6: расстояние Левенштейна (своя реализация, O(min(n,m)) памяти) и поиск опечаток."""

from collections import Counter


def levenshtein(a: str, b: str) -> int:
    """Классический ДП по строкам: храним только предыдущую строку -> O(min(n,m))."""
    if len(a) < len(b):
        a, b = b, a                       # короткая строка идёт в вектор
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def find_typo_pairs(tr_docs, top_k=800, max_dist=2):
    """Пары частотных слов корпуса с d=1 и d=2 — кандидаты на опечатки/формы."""
    freq = Counter(w.lower() for d in tr_docs for w in d
                   if w.isalpha() and len(w) > 2)
    words = [w for w, _ in freq.most_common(top_k)]
    pairs = {1: [], 2: []}
    for i in range(len(words)):
        wi = words[i]
        for j in range(i + 1, len(words)):
            wj = words[j]
            if abs(len(wi) - len(wj)) > max_dist:
                continue
            d = levenshtein(wi, wj)
            if d in (1, 2):
                pairs[d].append((wi, wj, freq[wi] * freq[wj]))
    for d in pairs:
        pairs[d].sort(key=lambda x: -x[2])
    print(f"[6] Пары слов с dist=1: {len(pairs[1])}, с dist=2: {len(pairs[2])} "
          f"(перебор по {top_k} самым частотным словам)")
    for d in (1, 2):
        print(f"    Топ-10 (dist={d}):")
        for w1, w2, score in pairs[d][:10]:
            print(f"      {w1} <-> {w2}  (freq {freq[w1]}x{freq[w2]})")
    return pairs, freq
