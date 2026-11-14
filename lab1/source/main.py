"""Точка входа лабораторной №1: uv run python -m lab1.source.main"""

from pathlib import Path

from lab1.source.clean import audit_noise, clean_and_tokenize
from lab1.source.data import eda, load_and_split
from lab1.source.levenshtein import find_typo_pairs
from lab1.source.lm import UNKN, NgramLM, generate, perplexity, prob_addk, prob_raw, prob_stupid, prob_unigram
from lab1.source.normalize import normalize_methods

OUT = Path(__file__).resolve().parents[1] / "output"


def main():
    train, test = load_and_split()
    eda(train)
    audit_noise(train)
    tr_docs, te_docs = clean_and_tokenize(train, test)
    normalize_methods(tr_docs)
    find_typo_pairs(tr_docs)

    lm = NgramLM(tr_docs, n=3)
    print(f"[7] Модель: N={lm.n}, словарь={len(lm.vocab)}, токенов train={lm.n_tokens}")

    te_docs_oov = [lm.map_oov(d) for d in te_docs]
    oov_tokens = sum(w == UNKN for d in te_docs_oov for w in d)
    print(f"    OOV-токенов [UNK] в test: {oov_tokens} "
          f"из {sum(len(d) for d in te_docs_oov)}")

    print("[8] Перплексия на test (база 2, по лог-правдоподобию):")
    rows = [
        ("unigram baseline (справка)", prob_unigram),
        ("raw MLE (без сглаживания)", prob_raw),
        ("add-k k=1.0 (Лаплас)", prob_addk),
        ("add-k k=0.1", lambda *a: prob_addk(*a, k=0.1)),
        ("add-k k=0.01", lambda *a: prob_addk(*a, k=0.01)),
        ("stupid backoff a=0.66", prob_stupid),
    ]
    for name, fn in rows:
        ppl, zeros, n = perplexity(lm, te_docs_oov, fn)
        print(f"    {name:<28}: PPL={ppl:12.2f}  (нулевых n-грамм: {zeros}/{n})")

    print("[9] Генерация (марковская цепь, max_len=30):")
    texts = generate(lm)
    for i, text in enumerate(texts, 1):
        print(f"    {i}. {text}")
    (OUT / "generated_texts.txt").write_text(
        "\n\n".join(f"{i}. {t}" for i, t in enumerate(texts, 1)), encoding="utf-8"
    )
    print(f"\nАртефакты: {OUT}")


if __name__ == "__main__":
    main()
