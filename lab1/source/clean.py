"""Пункты 3-4: аудит шума в домене и конвейер очистки регулярными выражениями."""

import html
import re

SS, ES = "<s>", "</s>"

NOISE_PATTERNS = {
    "html_tags": re.compile(r"</?(i|b|u|br|em|strong|a|small|cite)\b[^>]*>", re.I),
    "html_entities": re.compile(r"&(#?\w{2,8});"),
    "urls": re.compile(r"https?://\S+|www\.\S+"),
    "numbers_money": re.compile(r"\$?\d[\d,]*\.?\d*"),
    "emoticons": re.compile(r"[:;=8][-^]?[)(DPpd/3]|\b(lol|omg)\b", re.I),
    "smart_unicode": re.compile(r"[\u2018\u2019\u201c\u201d\u2013\u2014]"),
    "multi_punct": re.compile(r"[?!]{2,}|\.\.\.|…"),
}


def audit_noise(train):
    """Пункт 3: доказательства наличия типов шума — счётчиками по train."""
    print("[3] Аудит шума в корпусе (train):")
    stats = {}
    texts = train["Question"].astype(str)
    for name, pat in NOISE_PATTERNS.items():
        docs_hit = occ = 0
        for t in texts:
            found = pat.findall(t)
            if found:
                docs_hit += 1
                occ += len(found)
        stats[name] = (docs_hit, occ)
        share = 100.0 * docs_hit / len(train)
        print(f"    {name:>15}: в {docs_hit} документах ({share:.1f}%), вхождений {occ}")
    return stats


TAG_RE = re.compile(r"<[^>]+>")
# <a href="http://...">текст</a> -> [URL] текст: якорь вырезаем ДО удаления тегов,
# иначе URL внутри href будет уничтожен вместе с тегом.
ANCHOR_URL_RE = re.compile(r"<a\b[^>]*?https?://[^>]*?>(.*?)</a>", re.I | re.S)
URL_RE = re.compile(r"https?://[^\s\"'<>]+|www\.[^\s\"'<>]+")
NUM_RE = re.compile(r"\$\d[\d,]*\.?\d*|\b\d[\d,]*\.?\d*\b")
EMOT_RE = re.compile(r"[:;=8][-^]?[)(DPpd/3]")
MULTI_PUNCT_RE = re.compile(r"[?!]{2,}|\.\.\.|…")
SMART_MAP = {
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-",
}
WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Пункт 4: каждое правило — под свой тип шума; на выходе <s> ... </s>."""
    t = html.unescape(text)                      # HTML-сущности -> символы
    t = ANCHOR_URL_RE.sub(r" [URL] \1", t)       # <a href="http://..">текст</a> -> [URL] текст
    t = URL_RE.sub(" [URL] ", t)                 # URL -> токен [URL]
    t = TAG_RE.sub(" ", t)                       # оставшиеся HTML-теги -> пробел
    t = EMOT_RE.sub(" [EMOJI] ", t)              # смайлики -> токен [EMOJI]
    t = NUM_RE.sub(" [NUM] ", t)                 # числа и деньги -> [NUM]
    for k, v in SMART_MAP.items():               # smart-кавычки/тире -> ASCII
        t = t.replace(k, v)
    t = MULTI_PUNCT_RE.sub(" [ELLIPSIS] ", t)    # .../!!! -> маркер паузы
    t = re.sub(r"[^A-Za-z0-9\s'\"\-\[\]:.?!/]", " ", t)  # прочий мусор
    t = WS_RE.sub(" ", t).strip()
    return f"{SS} {t} {ES}"


def clean_and_tokenize(train, test):
    tr_docs = [clean_text(x).split() for x in train["Question"].astype(str)]
    te_docs = [clean_text(x).split() for x in test["Question"].astype(str)]
    print(f"[4] Очистка завершена; пример: {' '.join(tr_docs[0][:18])} ...")
    return tr_docs, te_docs
