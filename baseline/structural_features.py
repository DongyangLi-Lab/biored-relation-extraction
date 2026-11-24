"""Compute structural features for every candidate pair.

For each row of data/candidate_pairs.csv, read the original BioC JSON and compute:
    shares_sentence   1 if the two entities are mentioned in at least one shared sentence, else 0
    min_word_gap      fewest words between any mention of entity1 and any mention of entity2
    entity1_mentions  number of times entity1 is mentioned in the abstract
    entity2_mentions  number of times entity2 is mentioned in the abstract
    pair_type         copied from the candidate table (one-hot encoded later, at model time)

Values are raw. log1p, standardization, and one-hot encoding are fitted on Train inside the model pipeline.

Usage (from repository root):
    python baseline/structural_features.py
"""

import csv
import json
import re
from bisect import bisect_right
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "BioRED"
CANDIDATES = ROOT / "data" / "candidate_pairs.csv"
LABELS = ROOT / "data" / "relation_labels.csv"
OUTPUT = ROOT / "baseline" / "outputs" / "structural_features.csv"

KEY_COLUMNS = ["split", "pmid", "pair_type", "entity1", "entity2"]
FEATURE_COLUMNS = ["shares_sentence", "min_word_gap", "entity1_mentions", "entity2_mentions"]

# Same rule as same_sentence_rule.py: split after . ! ? when followed by space and an uppercase letter, digit, or bracket.
# Known issue: also splits after "vs." and variant notation such as "c. 1093". Waiting on a shared splitting method.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")
WORD = re.compile(r"\S+")


def sentence_spans(passage):
    """Return (start, end) document offsets of each sentence in one passage."""
    text, base = passage["text"], passage["offset"]
    spans, pos = [], 0
    for match in SENTENCE_END.finditer(text):
        spans.append((base + pos, base + match.start()))
        pos = match.end()
    spans.append((base + pos, base + len(text)))
    return spans


def word_starts(passage):
    """Return the document offset where each whitespace-separated word in one passage starts."""
    base = passage["offset"]
    return [base + match.start() for match in WORD.finditer(passage["text"])]


def parse_document(document):
    """Return {entity_id: [(sentence_index, first_word, last_word), ...]}, one tuple per mention."""
    sentences, starts = [], []
    for passage in document["passages"]:
        sentences += sentence_spans(passage)
        starts += word_starts(passage)

    def sentence_of(offset):
        for number, (start, end) in enumerate(sentences):
            if start <= offset < end:
                return number
        raise ValueError(f"PMID {document['id']}: offset {offset} is outside every sentence")

    def word_of(offset):
        return bisect_right(starts, offset) - 1

    mentions = {}
    for passage in document["passages"]:
        for annotation in passage["annotations"]:
            for location in annotation["locations"]:
                start = location["offset"]
                last = start + location["length"] - 1
                mention = (sentence_of(start), word_of(start), word_of(last))
                # One mention can carry several IDs, e.g. "3172,3651,6927". Count it for each.
                for entity_id in annotation["infons"]["identifier"].split(","):
                    mentions.setdefault(entity_id.strip(), []).append(mention)
    return mentions


def words_between(a, b):
    """Words strictly between two mentions; 0 if they touch or overlap."""
    _, a_first, a_last = a
    _, b_first, b_last = b
    return max(0, max(a_first, b_first) - min(a_last, b_last) - 1)


def pair_features(mentions1, mentions2):
    return {
        "shares_sentence": int(bool({m[0] for m in mentions1} & {m[0] for m in mentions2})),
        "min_word_gap": min(words_between(a, b) for a in mentions1 for b in mentions2),
        "entity1_mentions": len(mentions1),
        "entity2_mentions": len(mentions2),
    }


def load_mentions():
    """Return {pmid: {entity_id: mentions}} across all three splits."""
    documents = {}
    for split in ("Train", "Dev", "Test"):
        with open(RAW_DIR / f"{split}.BioC.JSON", encoding="utf-8") as f:
            for document in json.load(f)["documents"]:
                documents[document["id"]] = parse_document(document)
    return documents


def check_against_labels(rows):
    """Spot-check shares_sentence against same_sentence_mention for annotated pairs only."""
    computed = {
        (r["split"], r["pmid"], r["pair_type"], *sorted((r["entity1"], r["entity2"]))): r["shares_sentence"]
        for r in rows
    }
    agree = total = 0
    with open(LABELS, encoding="utf-8-sig") as f:
        for label in csv.DictReader(f):
            key = (label["split"], label["pmid"], label["pair_type"], *sorted((label["entity1"], label["entity2"])))
            total += 1
            agree += computed[key] == int(label["same_sentence_mention"] == "True")
    print(f"Spot check: shares_sentence matches same_sentence_mention on {agree}/{total} annotated pairs")


def main():
    documents = load_mentions()

    rows = []
    with open(CANDIDATES, encoding="utf-8-sig") as f:
        for candidate in csv.DictReader(f):
            mentions = documents[candidate["pmid"]]
            features = pair_features(mentions[candidate["entity1"]], mentions[candidate["entity2"]])
            rows.append({**{c: candidate[c] for c in KEY_COLUMNS}, **features})

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=KEY_COLUMNS + FEATURE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {OUTPUT.relative_to(ROOT)}")

    check_against_labels(rows)


if __name__ == "__main__":
    main()
