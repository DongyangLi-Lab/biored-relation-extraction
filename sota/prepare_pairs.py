"""Create shared BioRED inputs from the official BioC spans and fixed pair tables."""

import csv
import json
import re
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
KEY = ("split", "pmid", "pair_type", "entity1", "entity2")
SPLITS = ("Train", "Dev", "Test")
EXPECTED = {"Train": (21755, 4174), "Dev": (6334, 1161), "Test": (7047, 1163)}
RELATIONS = (
    "Association", "Positive_Correlation", "Negative_Correlation", "Bind",
    "Cotreatment", "Comparison", "Drug_Interaction", "Conversion",
)
BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
TOKEN = re.compile(r"\w+(?:[-\u2010\u2011]\w+)*")
MARKERS = ("ENTITY1", "ENTITY2")
TYPE_TEXT = {
    "GeneOrGeneProduct": "gene",
    "DiseaseOrPhenotypicFeature": "disease",
    "ChemicalEntity": "chemical",
    "SequenceVariant": "variant",
    "OrganismTaxon": "organism",
    "CellLine": "cell line",
}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def pair_key(row):
    return row["split"], row["pmid"], row["pair_type"], *sorted((row["entity1"], row["entity2"]))


def get_document(doc):
    pieces, ends, by_id = [], [], defaultdict(set)
    entity_types = {}
    cursor = 0
    for passage in doc["passages"]:
        text = passage["text"]
        pieces.append(text)
        ends.extend(cursor + m.start() for m in BOUNDARY.finditer(text))
        ends.append(cursor + len(text))
        for ann in passage.get("annotations", ()):
            ids = {s.strip() for s in ann["infons"]["identifier"].split(",")} - {"", "-"}
            for ident in ids:
                entity_type = ann["infons"]["type"]
                if entity_types.setdefault(ident, entity_type) != entity_type:
                    raise ValueError(f"Conflicting types for {ident} in {doc['id']}")
            for loc in ann["locations"]:
                start = loc["offset"] - passage["offset"]
                end = start + loc["length"]
                if not 0 <= start < end <= len(text):
                    raise ValueError(f"Invalid BioC span in {doc['id']}: {loc}")
                for ident in ids:
                    by_id[ident].add((cursor + start, cursor + end))
        cursor += len(text) + 1
    text = "\n".join(pieces)
    tokens = list(TOKEN.finditer(text))
    return (
        text, ends, sorted(m.start() for m in tokens),
        sorted(m.end() for m in tokens),
        {ident: sorted(spans) for ident, spans in by_id.items()},
        entity_types,
    )


def mask_region(text, start, stop, a_spans, b_spans):
    spans = sorted(
        (a, b, role)
        for role, mentions in enumerate((a_spans, b_spans))
        for a, b in mentions
        if start <= a < b <= stop
    )
    chunks, cursor, i = [], start, 0
    while i < len(spans):
        a, b, role = spans[i]
        roles = {role}
        i += 1
        while i < len(spans) and spans[i][0] < b:
            b = max(b, spans[i][1])
            roles.add(spans[i][2])
            i += 1
        chunks.extend((text[cursor:a], " ", " ".join(MARKERS[r] for r in sorted(roles)), " "))
        cursor = b
    chunks.append(text[cursor:stop])
    return " ".join("".join(chunks).split())


def prepare_pair(doc, key):
    text, sent_ends, token_starts, token_ends, mentions, entity_types = doc
    a_spans, b_spans = mentions[key[3]], mentions[key[4]]

    def nearest_rank(pair):
        first, second = sorted(pair)
        gap = max(0, bisect_left(token_starts, second[0]) - bisect_right(token_ends, first[1]))
        return gap, max(0, second[0] - first[1]), first[0], second[0]

    chosen = min(((a, b) for a in a_spans for b in b_spans), key=nearest_rank)
    a_sent, b_sent = (bisect_right(sent_ends, s[0]) for s in chosen)
    pieces = []
    for sid in sorted({a_sent, b_sent}):
        chosen_here = [s for s in chosen if bisect_right(sent_ends, s[0]) == sid]
        start = sent_ends[sid - 1] if sid else 0
        stop = max(sent_ends[sid], *(s[1] for s in chosen_here))
        pieces.append(mask_region(text, start, stop, a_spans, b_spans))
    local = " [SEP] ".join(pieces)
    abstract = mask_region(text, 0, len(text), a_spans, b_spans)
    if any(not all(marker in view.split() for marker in MARKERS) for view in (local, abstract)):
        raise ValueError(f"Missing endpoint marker in {key}")

    def sentence_ids(spans):
        return {bisect_right(sent_ends, start) for start, _ in spans}

    return {
        "annotated_relation": None,
        "relation_type": None,
        "same_sentence": int(bool(sentence_ids(a_spans) & sentence_ids(b_spans))),
        "min_token_gap": nearest_rank(chosen)[0],
        "entity1_mentions": len(a_spans),
        "entity2_mentions": len(b_spans),
        "entity1_type": TYPE_TEXT[entity_types[key[3]]],
        "entity2_type": TYPE_TEXT[entity_types[key[4]]],
        "entity1_name": " ".join(text[chosen[0][0]:chosen[0][1]].split()),
        "entity2_name": " ".join(text[chosen[1][0]:chosen[1][1]].split()),
        "context_local": local,
        "context_abstract": abstract,
    }


def main():
    candidates = read_csv(DATA / "candidate_pairs.csv")
    relations = {pair_key(r): r["relation_type"] for r in read_csv(DATA / "relation_labels.csv")}
    keys = {pair_key(row) for row in candidates}
    positives = {pair_key(row) for row in candidates if row["annotated_relation"] == "1"}
    if len(keys) != len(candidates) or positives != relations.keys() or set(relations.values()) != set(RELATIONS):
        raise ValueError("Candidate keys or labels are inconsistent with the relation table")
    output = DATA / "processed"
    output.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split in SPLITS:
        raw = DATA / "raw" / "BioRED" / f"{split}.BioC.JSON"
        docs = {
            d["id"]: get_document(d)
            for d in json.loads(raw.read_text(encoding="utf-8"))["documents"]
        }
        rows = []
        for row in candidates:
            if row["split"] != split:
                continue
            key = pair_key(row)
            features = prepare_pair(docs[key[1]], key)
            features["annotated_relation"] = int(row["annotated_relation"])
            features["relation_type"] = relations.get(key, "")
            rows.append(dict(zip(KEY, key), **features))
        labels = Counter(r["relation_type"] for r in rows if r["annotated_relation"])
        if (len(rows), sum(labels.values())) != EXPECTED[split]:
            raise ValueError(f"Unexpected counts in {split}")
        target = output / f"{split}.csv"
        temporary = target.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(target)
        stats[split] = {
            "pairs": len(rows),
            "annotated": sum(labels.values()),
            "same_sentence": sum(r["same_sentence"] for r in rows),
            "annotated_same_sentence": sum(r["same_sentence"] for r in rows if r["annotated_relation"]),
            "relation_types": dict(labels),
        }
        print(split, stats[split], flush=True)
    (output / "summary.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
