from collections.abc import Sequence

import numpy as np
import torch

from .alignment import IGNORE_INDEX, levenshtein_alignment
from .phonemes import split_phones


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            old = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = old
    return dp[m]


def token_errors_against_canonical(canonical: Sequence[str], other: Sequence[str]) -> dict[int, str]:
    errors = {}
    align = levenshtein_alignment(canonical, other)
    pos = -1
    for op, _, tgt in align:
        if op == "insert":
            errors[pos] = tgt
            continue
        pos += 1
        if op == "substitute":
            errors[pos] = tgt
        elif op == "delete":
            errors[pos] = "<deleted>"
    return errors


def mdd_metrics_from_sequences(canonicals: Sequence[str], transcripts: Sequence[str], predicts: Sequence[str]) -> dict:
    tp = fp = fn = 0
    correct_diag = wrong_diag = 0
    per_values = []
    copy_count = 0
    pred_edit_counts = []

    for canonical, transcript, predict in zip(canonicals, transcripts, predicts):
        c = split_phones(canonical)
        t = split_phones(transcript)
        p = split_phones(predict)
        actual_errors = token_errors_against_canonical(c, t)
        pred_errors = token_errors_against_canonical(c, p)

        actual_pos = set(actual_errors)
        pred_pos = set(pred_errors)
        tp += len(actual_pos & pred_pos)
        fp += len(pred_pos - actual_pos)
        fn += len(actual_pos - pred_pos)
        for pos in actual_pos & pred_pos:
            if pred_errors[pos] == actual_errors[pos]:
                correct_diag += 1
            else:
                wrong_diag += 1

        per_values.append(edit_distance(p, t) / max(1, len(t)))
        copy_count += int(p == c)
        pred_edit_counts.append(len(pred_pos))

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    der = wrong_diag / max(1, correct_diag + wrong_diag)
    per = float(np.mean(per_values)) if per_values else 0.0
    score = 0.5 * f1 + 0.4 * (1 - der) + 0.1 * (1 - per)
    return {
        "F1": f1,
        "DER": der,
        "PER": per,
        "Score": score,
        "precision": precision,
        "recall": recall,
        "true_reject": tp,
        "false_reject": fp,
        "false_accept": fn,
        "correct_diagnosis": correct_diag,
        "wrong_diagnosis": wrong_diag,
        "canonical_copy_rate": copy_count / max(1, len(canonicals)),
        "avg_predicted_edits_per_utterance": float(np.mean(pred_edit_counts)) if pred_edit_counts else 0.0,
        "zero_edit_prediction_rate": float(np.mean([x == 0 for x in pred_edit_counts])) if pred_edit_counts else 0.0,
    }


def token_supervised_metrics(outputs: dict, batch: dict) -> dict:
    det_pred = outputs["detection_logits"].argmax(dim=-1)
    op_pred = outputs["operation_logits"].argmax(dim=-1)
    repl_pred = outputs["replacement_logits"].argmax(dim=-1)

    det_target = batch["detection_labels"]
    op_target = batch["operation_labels"]
    repl_target = batch["replacement_targets"]
    valid_det = det_target != IGNORE_INDEX
    valid_op = op_target != IGNORE_INDEX
    valid_repl = repl_target != IGNORE_INDEX

    tp = int(((det_pred == 1) & (det_target == 1) & valid_det).sum().detach().cpu())
    fp = int(((det_pred == 1) & (det_target == 0) & valid_det).sum().detach().cpu())
    fn = int(((det_pred == 0) & (det_target == 1) & valid_det).sum().detach().cpu())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    op_acc = float(((op_pred == op_target) & valid_op).sum().detach().cpu()) / max(1, int(valid_op.sum().detach().cpu()))
    repl_acc = float(((repl_pred == repl_target) & valid_repl).sum().detach().cpu()) / max(1, int(valid_repl.sum().detach().cpu()))
    utt_pred = (torch.sigmoid(outputs["utterance_logits"]) >= 0.5).long()
    utt_target = batch["utterance_labels"].long()
    utt_tp = int(((utt_pred == 1) & (utt_target == 1)).sum().detach().cpu())
    utt_fp = int(((utt_pred == 1) & (utt_target == 0)).sum().detach().cpu())
    utt_fn = int(((utt_pred == 0) & (utt_target == 1)).sum().detach().cpu())
    utt_p = utt_tp / max(1, utt_tp + utt_fp)
    utt_r = utt_tp / max(1, utt_tp + utt_fn)
    utt_f1 = 2 * utt_p * utt_r / max(1e-12, utt_p + utt_r)
    return {
        "token_detection_f1": f1,
        "token_detection_precision": precision,
        "token_detection_recall": recall,
        "operation_accuracy": op_acc,
        "replacement_accuracy_on_sub": repl_acc,
        "utterance_f1": utt_f1,
    }
