from collections.abc import Sequence

import pandas as pd

from .phonemes import split_phones


OP_KEEP = 0
OP_SUB = 1
OP_NAMES = {OP_KEEP: "KEEP", OP_SUB: "SUBSTITUTE"}
IGNORE_INDEX = -100


def levenshtein_alignment(src: Sequence[str], tgt: Sequence[str]) -> list[tuple[str, str | None, str | None]]:
    n, m = len(src), len(tgt)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "delete"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "insert"

    order = {"equal": 0, "substitute": 1, "delete": 2, "insert": 3}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if src[i - 1] == tgt[j - 1]:
                best = (dp[i - 1][j - 1], "equal")
            else:
                best = (dp[i - 1][j - 1] + 1, "substitute")
            candidates = [best, (dp[i - 1][j] + 1, "delete"), (dp[i][j - 1] + 1, "insert")]
            dp[i][j], back[i][j] = min(candidates, key=lambda x: (x[0], order[x[1]]))

    align = []
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "equal":
            align.append(("equal", src[i - 1], tgt[j - 1]))
            i -= 1
            j -= 1
        elif op == "substitute":
            align.append(("substitute", src[i - 1], tgt[j - 1]))
            i -= 1
            j -= 1
        elif op == "delete":
            align.append(("delete", src[i - 1], None))
            i -= 1
        elif op == "insert":
            align.append(("insert", None, tgt[j - 1]))
            j -= 1
        else:
            raise RuntimeError("Invalid alignment backtrace.")
    align.reverse()
    return align


def make_v1_label_row(row: pd.Series, token_to_id: dict[str, int], unk_id: int) -> dict:
    canonical_tokens = split_phones(row["canonical"])
    transcript_tokens = split_phones(row["transcript"])
    align = levenshtein_alignment(canonical_tokens, transcript_tokens)

    canonical_ids = []
    operation_labels = []
    detection_labels = []
    replacement_targets = []
    ignored_insertions = []
    ignored_deletions = []

    canon_pos = 0
    for op, src_tok, tgt_tok in align:
        if op == "insert":
            ignored_insertions.append({"gap_after": canon_pos - 1, "token": tgt_tok})
            continue

        canonical_ids.append(token_to_id.get(src_tok, unk_id))
        if op == "equal":
            operation_labels.append(OP_KEEP)
            detection_labels.append(0)
            replacement_targets.append(IGNORE_INDEX)
        elif op == "substitute":
            operation_labels.append(OP_SUB)
            detection_labels.append(1)
            replacement_targets.append(token_to_id.get(tgt_tok, unk_id))
        elif op == "delete":
            operation_labels.append(IGNORE_INDEX)
            detection_labels.append(IGNORE_INDEX)
            replacement_targets.append(IGNORE_INDEX)
            ignored_deletions.append({"position": canon_pos, "token": src_tok})
        else:
            raise ValueError(op)
        canon_pos += 1

    if len(canonical_ids) != len(canonical_tokens):
        raise AssertionError("label length must match canonical token length")

    return {
        "canonical_ids": canonical_ids,
        "operation_labels": operation_labels,
        "detection_labels": detection_labels,
        "replacement_targets": replacement_targets,
        "utterance_label": int(any(x == 1 for x in detection_labels)),
        "ignored_insertions": ignored_insertions,
        "ignored_deletions": ignored_deletions,
    }
