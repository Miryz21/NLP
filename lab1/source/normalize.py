"""Пункт 5: сравнительная нормализация — стемминг против лемматизации."""

from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag

WN_POS = {"N": wordnet.NOUN, "V": wordnet.VERB, "A": wordnet.ADJ, "R": wordnet.ADV}


def normalize_methods(tr_docs):
    """Сравнение размеров словарей: raw / stemming / lemmatization."""
    stemmer, lemmatizer = PorterStemmer(), WordNetLemmatizer()
    raw_vocab, stem_vocab, lemm_vocab = set(), set(), set()
    for doc in tr_docs:
        low = [w.lower() for w in doc if not w.startswith("[") and w not in ("<s>", "</s>")]
        raw_vocab.update(low)
        stem_vocab.update(stemmer.stem(w) for w in low)
        tagged = pos_tag(low)
        lemm_vocab.update(
            lemmatizer.lemmatize(w, WN_POS[tag[0][0]]) if tag[0][0] in WN_POS
            else lemmatizer.lemmatize(w)
            for w, tag in tagged
        )
    print("[5] Нормализация (уникальные токены словаря):")
    print(f"    без нормализации : {len(raw_vocab)}")
    print(f"    PorterStemmer    : {len(stem_vocab)} "
          f"(-{100 * (1 - len(stem_vocab) / len(raw_vocab)):.1f}%)")
    print(f"    WordNetLemmatizer: {len(lemm_vocab)} "
          f"(-{100 * (1 - len(lemm_vocab) / len(raw_vocab)):.1f}%)")
    return raw_vocab, stem_vocab, lemm_vocab
