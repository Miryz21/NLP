"""Пункты 7-9: N-граммная модель, сглаживание, перплексия, генерация."""

import math
import random
from collections import Counter, defaultdict

SS, ES = "<s>", "</s>"
UNKN = "[UNK]"


class NgramLM:
    """Явная триграммная модель: счётчики uni/bi/tri + словарь с [UNK]."""

    def __init__(self, docs, n=3):
        assert n == 3
        self.n = n
        self.uni = Counter()
        self.bi = defaultdict(Counter)
        self.tri = defaultdict(Counter)
        for doc in docs:
            for w in doc:
                self.uni[w] += 1
            for w1, w2 in zip(doc, doc[1:]):
                self.bi[w1][w2] += 1
            for w1, w2, w3 in zip(doc, doc[1:], doc[2:]):
                self.tri[(w1, w2)][w3] += 1
        self.n_tokens = sum(self.uni.values())
        self.vocab = set(self.uni) | {UNKN}
        self.uni[UNKN] += 1

    def map_oov(self, doc):
        """Неизвестные слова тестовой выборки -> токен [UNK]."""
        return [w if w in self.vocab else UNKN for w in doc]


def prob_raw(lm, w1, w2, w3):
    """Строгое MLE без сглаживания: P(w3|w1w2)=c(w1w2w3)/c(w1w2).
    Не встреченный в train контекст даёт вероятность 0 -> обнуляет правдоподобие."""
    c2 = sum(lm.tri[(w1, w2)].values())
    return lm.tri[(w1, w2)][w3] / c2 if c2 else 0.0


def prob_addk(lm, w1, w2, w3, k=1.0):
    """Лапласово add-k сглаживание на триграммном уровне."""
    ctx = lm.tri[(w1, w2)]
    return (ctx[w3] + k) / (sum(ctx.values()) + k * len(lm.vocab))


def prob_stupid(lm, w1, w2, w3, alpha=0.66):
    """Stupid Backoff (Neyman et al., 2004), рекурсивно:
    P(w3|w1w2) = c(w1w2w3)>0 ? c/c2 : alpha*P(w3|w2);  P(w3|w2) = c>0 ? c/c1 : alpha*P_uni(w3).
    Zero-вероятности не остаётся — backoff спускается до unigram."""
    ctx = lm.tri[(w1, w2)]
    c3 = ctx[w3]
    if c3:
        return c3 / sum(ctx.values())
    b = lm.bi[w2]
    c1 = sum(b.values())
    if b[w3]:
        return alpha * b[w3] / c1
    return alpha * alpha * lm.uni[w3] / lm.n_tokens


def prob_unigram(lm, w1, w2, w3):
    """Бейслайн-базы: P(w3) = c(w3)/N. Перплексия ~ effective vocabulary."""
    return lm.uni[w3] / lm.n_tokens


def perplexity(lm, docs, prob_fn):
    """PPL = 2^(-1/N * sum log2 P) — считается через лог-правдоподобие."""
    total_log, n, zero_hits = 0.0, 0, 0
    for doc in docs:
        for w1, w2, w3 in zip(doc, doc[1:], doc[2:]):
            p = prob_fn(lm, w1, w2, w3)
            if p <= 0:
                zero_hits += 1
                total_log += math.log2(1e-10)
            else:
                total_log += math.log2(p)
            n += 1
    return 2.0 ** (-total_log / n), zero_hits, n


def generate(lm, n_texts=5, max_len=30, seed=42):
    """Марковская генерация по триграммам со stupid backoff; старт <s>, стоп </s>."""
    rng = random.Random(seed)
    choices = sorted(lm.vocab - {SS, ES, UNKN})
    texts = []
    for _ in range(n_texts):
        w1 = w2 = SS
        seq = [SS, SS]
        for _ in range(max_len):
            ctx = lm.tri[(w1, w2)]
            weights = [ctx[w] for w in choices] if sum(ctx.values()) else None
            if weights is None:
                b = lm.bi[w2]
                weights = [b[w] for w in choices]
            s = sum(weights)
            if s == 0:
                weights = [lm.uni[w] for w in choices]
                s = sum(weights)
            w3 = rng.choices(choices, weights=weights, k=1)[0]
            seq.append(w3)
            if w3 == ES:
                break
            w1, w2 = w2, w3
        texts.append(" ".join(w for w in seq if w not in (SS, ES)))
    return texts
