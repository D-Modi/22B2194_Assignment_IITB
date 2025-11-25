import json
import random
from collections import defaultdict
import re

def read_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def write_jsonl(path, items):
    with open(path, "w") as f:
        for obj in items:
            f.write(json.dumps(obj) + "\n")

train_items = read_jsonl("pii_ner_assignment_IITB/data/test.jsonl")
print("Existing train size:", len(train_items))

# Collect all entity text values by label from existing train
entities_by_label = defaultdict(list)

for ex in train_items:
    text = ex["text"]
    for ent in ex.get("entities", []):
        s, e, label = ent["start"], ent["end"], ent["label"]
        value = text[s:e]
        # Basic sanity: avoid empty or whitespace-only
        if value.strip():
            entities_by_label[label].append(value.strip())

for label, vals in entities_by_label.items():
    print(label, "->", len(vals), "examples")
    print(label, "->", vals[:5], "examples")
    


def extract_template(text, entities):
    """
    text: original utterance string
    entities: list of {"start": s, "end": e, "label": L}
    returns: template string like "my name is {PERSON_NAME} from {CITY}"
    """
    if not entities:
        return None
    
    ents_sorted = sorted(entities, key=lambda x: x["start"])
    template_parts = []
    cur = 0
    
    for ent in ents_sorted:
        s, e, label = ent["start"], ent["end"], ent["label"]
        template_parts.append(text[cur:s])
        template_parts.append("{" + label + "}")
        cur = e
    
    template_parts.append(text[cur:])
    template = "".join(template_parts)
    return template

templates = []

for ex in train_items:
    text = ex["text"]
    ents = ex.get("entities", [])
    tmpl = extract_template(text, ents)
    if tmpl is not None:
        templates.append(tmpl)

print("Extracted templates:", len(templates))
print("Sample templates:")
for t in templates[:5]:
    print("-", t)
    
    
def instantiate_template(template, entities_by_label):
    """
    template: string with placeholders like "... {PERSON_NAME} ..."
    entities_by_label: dict label -> list of possible values
    returns: example dict with text + entities, or None if a label has no values
    """
    out_text = ""
    entities = []
    i = 0
    
    while i < len(template):
        if template[i] == "{":
            j = template.find("}", i)
            if j == -1:
                # malformed template
                return None
            label = template[i+1:j]
            if label not in entities_by_label or not entities_by_label[label]:
                return None
            value = random.choice(entities_by_label[label])
            
            start = len(out_text)
            out_text += value
            end = len(out_text)
            
            entities.append({
                "start": start,
                "end": end,
                "label": label
            })
            next_idx = j + 1
            if next_idx < len(template):
                next_char = template[next_idx]
                # if next char is not whitespace or punctuation, insert a space
               # if not next_char.isspace() and next_char not in ".,!?;:":
                out_text += " "
            i = j + 1
        else:
            out_text += template[i]
            i += 1
    
    if not entities:
        return None
    
    example = {
        "id": f"synthetic_{random.randint(100000, 999999)}",
        "text": out_text,
        "entities": entities
    }
    return example


def drop_punctuation(text):
    return re.sub(r"[.,!?]", "", text)

def lowercase(text):
    return text.lower()

def repeat_word(text):
    words = text.split()
    if len(words) < 2:
        return text
    idx = random.randint(0, len(words)-1)
    words.insert(idx, words[idx])
    return " ".join(words)

def insert_filler(text):
    fillers = ["uh", "umm", "you know", "like", "basically"]
    words = text.split()
    if not words:
        return text
    idx = random.randint(0, len(words)-1)
    words.insert(idx, random.choice(fillers))
    return " ".join(words)

def substitute_homophones(text):
    homophones = {
        " to ": " two ",
        " for ": " four ",
        " email ": " male ",
        " card ": " car ",
        " at ": " at the rate ",
        " from ": " frum ",
    }
    # pad with spaces at start/end to simplify replacements
    t = " " + text + " "
    for k, v in homophones.items():
        t = t.replace(k, " " + v + " ")
    return t.strip()

def typo_noise(text, prob=0.03):
    # small random char substitutions
    result = []
    for ch in text:
        if random.random() < prob:
            result.append(random.choice("abcdefghijklmnopqrstuvwxyz "))
        else:
            result.append(ch)
    return "".join(result)

def apply_noise_to_segment(seg):
    """
    Apply 1-3 random noise ops to a non-entity text segment.
    """
    if not seg.strip():
        return seg
    
    transforms = [
        drop_punctuation,
        lowercase,
        repeat_word,
        insert_filler,
        substitute_homophones,
        typo_noise
    ]
    
    num_ops = random.randint(1, 3)
    ops = random.sample(transforms, num_ops)
    
    out = seg
    for op in ops:
        out = op(out)
    return out

def add_context_noise(example):
    """
    Given example with text + entities (correct spans),
    return a new example where only the context (non-entity parts)
    is noised, entities are unchanged but spans recomputed.
    """
    text = example["text"]
    ents = sorted(example["entities"], key=lambda x: x["start"])
    
    segments = []
    cur = 0
    
    for ent in ents:
        s, e, label = ent["start"], ent["end"], ent["label"]
        # non-entity segment
        if cur < s:
            segments.append({"type": "O", "text": text[cur:s]})
        # entity segment
        segments.append({"type": "E", "text": text[s:e], "label": label})
        cur = e
    # tail context
    if cur < len(text):
        segments.append({"type": "O", "text": text[cur:]})
    
    # apply noise to O segments
    new_text = ""
    new_entities = []
    for seg in segments:
        if seg["type"] == "O":
            noisy = apply_noise_to_segment(seg["text"])
            new_text += noisy
        else:
            # entity: keep text as is
            value = seg["text"]
            if new_text and not new_text[0].isspace():
                new_text = " " + new_text
            if new_text and not new_text[-1].isspace():
                new_text += " "
            start = len(new_text)
            new_text += value
            end = len(new_text)
            new_entities.append({
                "start": start,
                "end": end,
                "label": seg["label"]
            })
    
    return {
        "id": example["id"] + "_noisy",
        "text": new_text,
        "entities": new_entities
    }


def generate_synthetic_noisy(
    n=800,
    templates=None,
    entities_by_label=None
):
    examples = []
    attempts = 0
    max_attempts = n * 5  # avoid infinite loops if many templates invalid
    
    while len(examples) < n and attempts < max_attempts:
        attempts += 1
        tmpl = random.choice(templates)
        base_ex = instantiate_template(tmpl, entities_by_label)
        if base_ex is None:
            continue
        noisy_ex = add_context_noise(base_ex)
        examples.append(noisy_ex)
    
    return examples

synthetic_noisy_examples = generate_synthetic_noisy(
    n=800,
    templates=templates,
    entities_by_label=entities_by_label
)

print("Generated synthetic noisy examples:", len(synthetic_noisy_examples))
print("Sample synthetic example:")
print(synthetic_noisy_examples[:10])

train_path = "pii_ner_assignment_IITB/data/test.jsonl"
train_extended = train_items + synthetic_noisy_examples
write_jsonl(train_path, train_extended)
print("New train size:", len(train_extended))



dev_path = "pii_ner_assignment_IITB/data/dev.jsonl"
dev_items = read_jsonl(dev_path)
print("Existing dev size:", len(dev_items))

# 2. Generate synthetic noisy dev examples
#    (aim for ~100–200, as per assignment)
dev_synth = generate_synthetic_noisy(
    n=150,                      # choose 100–200
    templates=templates,        # from train
    entities_by_label=entities_by_label
)

print("Generated synthetic dev examples:", len(dev_synth))
print("Sample synthetic dev example:")
print(json.dumps(dev_synth[0], indent=2))

# 3. Merge and overwrite dev.jsonl
dev_extended = dev_items + dev_synth
write_jsonl(dev_path, dev_extended)

print("New dev size:", len(dev_extended))





    
    
