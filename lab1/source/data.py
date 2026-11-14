"""Пункты 1-2: загрузка датасета Jeopardy, стратифицированный сплит, EDA."""

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "datasets" / "jeopardy.csv"
OUT = BASE / "lab1" / "output"
OUT.mkdir(exist_ok=True)


def load_and_split(test_size=0.2, seed=42):
    """Пункт 1: целевая колонка Question, сплит 80/20 со стратификацией по Round."""
    df = pd.read_csv(DATA)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["Question"]).reset_index(drop=True)
    # Нативной бинарной метки у датасета нет; Round — естественная страта,
    # задающая распределение длины и стиля вопросов.
    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["Round"].astype(str)
    )
    print(f"[1] Всего документов: {len(df)}")
    print(f"    Train: {len(train)}  Test: {len(test)}  (80/20, stratify=Round)")
    print(f"    Классы Round (train): {dict(Counter(train['Round'].astype(str)))}")
    return train, test


def eda(train):
    """Пункт 2: структура корпуса на train + графики длин и топ-30 стоп-слов."""
    texts = train["Question"].astype(str)
    n_words = [len(word_tokenize(t)) for t in texts]
    n_chars = [len(t) for t in texts]
    print(f"[2] EDA (train): документов = {len(texts)}")
    print(f"    Средняя длина: {sum(n_words)/len(n_words):.2f} слов, "
          f"{sum(n_chars)/len(n_chars):.2f} символов")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(n_words, bins=60, color="steelblue")
    axes[0].set_title("Распределение длины (слова), train")
    axes[0].set_xlabel("слов")
    axes[1].hist(n_chars, bins=60, color="darkseagreen")
    axes[1].set_title("Распределение длины (символы), train")
    axes[1].set_xlabel("символов")
    fig.tight_layout()
    fig.savefig(OUT / "lengths.png", dpi=120)
    plt.close(fig)

    sw = set(stopwords.words("english"))
    cnt = Counter()
    for t in texts:
        cnt.update(w.lower() for w in word_tokenize(t) if w.lower() in sw)
    top = cnt.most_common(30)
    fig = plt.figure(figsize=(10, 6))
    words = [w for w, _ in top][::-1]
    counts = [c for _, c in top][::-1]
    plt.bar(words, counts, color="indianred")
    plt.title("Top-30 стоп-слов, train")
    plt.tight_layout()
    fig.savefig(OUT / "stopwords.png", dpi=120)
    plt.close(fig)
    print(f"    Топ-10 стоп-слов: {top[:10]}")
