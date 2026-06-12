from collections.abc import Iterable, Sequence

import pandas as pd


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN]


def split_phones(text) -> list[str]:
    if pd.isna(text):
        return []
    return str(text).strip().split()


def iter_phone_tokens_from_df(df: pd.DataFrame, columns: Sequence[str]) -> Iterable[str]:
    for column in columns:
        if column not in df.columns:
            continue
        for text in df[column].fillna(""):
            yield from split_phones(text)


def build_phoneme_vocab(*frames: pd.DataFrame) -> dict:
    vocab_tokens = set()
    for frame in frames:
        vocab_tokens.update(iter_phone_tokens_from_df(frame, ["canonical", "transcript"]))
    id_to_token = SPECIAL_TOKENS + sorted(vocab_tokens)
    token_to_id = {token: idx for idx, token in enumerate(id_to_token)}
    return {
        "id_to_token": id_to_token,
        "token_to_id": token_to_id,
        "pad_id": token_to_id[PAD_TOKEN],
        "unk_id": token_to_id[UNK_TOKEN],
    }
