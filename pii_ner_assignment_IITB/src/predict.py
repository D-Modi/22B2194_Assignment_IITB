
import json
import argparse
import os
import re

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, LABEL2ID, label_is_pii


# Threshold: PII tokens below this prob will be forced to "O"
PII_TOKEN_THRESHOLD = 0.80


def bio_to_spans(text, offsets, label_ids):
    """
    Convert token-level BIO labels into character-level spans.

    Args:
        text: original utterance string
        offsets: list of (start_char, end_char) from tokenizer
        label_ids: list of label IDs (one per token)

    Returns:
        List of (start, end, label_str) spans.
    """
    spans = []
    current_label = None
    current_start = None
    current_end = None

    def close_span():
        nonlocal current_label, current_start, current_end
        if current_label is not None and current_start is not None and current_end is not None:
            if current_start < current_end:
                spans.append((current_start, current_end, current_label))
        current_label = None
        current_start = None
        current_end = None

    for (start, end), lid in zip(offsets, label_ids):
        # Skip special tokens [CLS], [SEP] etc. which typically have (0,0)
        if start == 0 and end == 0:
            continue
        if start is None or end is None or start >= end:
            continue

        label = ID2LABEL.get(int(lid), "O")

        if label == "O":
            close_span()
            continue

        # BIO parsing
        if "-" in label:
            prefix, ent_type = label.split("-", 1)
        else:
            prefix, ent_type = "O", None

        if prefix == "B":
            # Start of new span
            close_span()
            current_label = ent_type
            current_start = start
            current_end = end

        elif prefix == "I":
            if current_label == ent_type:
                # Continuation
                current_end = end
            else:
                # Illegal I-X without correct B-X; start new span for robustness
                close_span()
                current_label = ent_type
                current_start = start
                current_end = end
        else:
            # Anything else treated as O
            close_span()

    close_span()
    return spans


# ------------------------ #
#  Plausibility functions  #
# ------------------------ #

NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten"
}

MONTH_WORDS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
}


def is_plausible_credit_card(span_text: str) -> bool:
    # Strip non-digit characters
    digits = re.sub(r"[^\d]", "", span_text)
    # Typical card numbers are 13-19 digits
    return digits.isdigit() and 13 <= len(digits) <= 19


def is_plausible_email(span_text: str) -> bool:
    s = span_text.strip().lower()

    # Standard email pattern
    if "@" in s:
        # Rough, but good enough: something@something.something
        return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", s))

    # ASR-style: "name at gmail dot com"
    if " at " in s and " dot " in s:
        return True

    return False


def is_plausible_phone(span_text: str) -> bool:
    s = span_text.lower()
    # Count digits
    digits = re.sub(r"[^\d]", "", s)
    # Count number words
    words = re.findall(r"[a-z]+", s)
    num_word_count = sum(1 for w in words if w in NUMBER_WORDS)

    # Either enough digits, or enough number-words
    if len(digits) >= 8:
        return True
    if num_word_count >= 5:
        return True

    return False


def is_plausible_date(span_text: str) -> bool:
    s = span_text.lower().strip()

    # Numeric date patterns
    if re.search(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", s):
        return True

    # "24 january 2025", "20 november 2026", etc.
    tokens = s.split()
    has_month = any(t in MONTH_WORDS for t in tokens)
    has_digit = any(any(c.isdigit() for c in t) for t in tokens)
    if has_month and has_digit:
        return True

    return False


def is_plausible_person(span_text: str) -> bool:
    # Very simple: length > 2 and at least 2 characters
    s = span_text.strip()
    # Filter extremely short spans like "i", "a"
    if len(s) < 2:
        return False
    return True


def postprocess_spans(text, spans):
    """
    Apply post-processing to increase PII precision.

    Args:
        text: original utterance string
        spans: list of (start, end, label_str) from BIO decoding

    Returns:
        Filtered list of spans with same structure.
    """
    filtered = []
    for (s, e, lab) in spans:
        span_text = text[s:e]

        # Basic sanity: drop extremely short spans
        if len(span_text.strip()) == 0:
            continue

        # Apply label-specific plausibility filters only to PII
        if label_is_pii(lab):
            # Generic minimum length for PII to avoid junk
            if len(span_text.strip()) < 3:
                continue

            if lab == "CREDIT_CARD" and not is_plausible_credit_card(span_text):
                continue

            if lab == "EMAIL" and not is_plausible_email(span_text):
                continue

            if lab == "PHONE" and not is_plausible_phone(span_text):
                continue

            if lab == "DATE" and not is_plausible_date(span_text):
                continue

            if lab == "PERSON_NAME" and not is_plausible_person(span_text):
                continue

        # Non-PII (CITY, LOCATION) we keep as-is
        filtered.append((s, e, lab))

    return filtered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir if args.model_name is None else args.model_name
    )
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    results = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0]  # [seq_len, num_labels]

                # Softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)
                max_probs, pred_ids = probs.max(dim=-1)

                pred_ids = pred_ids.cpu().tolist()
                max_probs = max_probs.cpu().tolist()

            # 1) Probability-based filtering: force low-confidence PII tokens to "O"
            for i, (lid, p) in enumerate(zip(pred_ids, max_probs)):
                label_str = ID2LABEL.get(int(lid), "O")
                if label_str != "O" and label_is_pii(label_str) and p < PII_TOKEN_THRESHOLD:
                    pred_ids[i] = LABEL2ID["O"]

            # 2) BIO -> spans
            spans = bio_to_spans(text, offsets, pred_ids)

            # 3) Span-level post-processing / plausibility filtering
            spans = postprocess_spans(text, spans)

            ents = []
            for s, e, lab in spans:
                ents.append(
                    {
                        "start": int(s),
                        "end": int(e),
                        "label": lab,
                        "pii": bool(label_is_pii(lab)),
                    }
                )
            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()

