import os
from collections import Counter, defaultdict

def count_split(root):
    # root should point to the dataset folder that has train/val/test subfolders
    splits = ["train", "val", "test"]
    counts = {s: Counter() for s in splits}
    for s in splits:
        split_dir = os.path.join(root, s)
        if not os.path.isdir(split_dir):
            continue
        for cls in sorted(d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))):
            cls_dir = os.path.join(split_dir, cls)
            n = sum(1 for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f)))
            counts[s][cls] += n
    return counts

def print_counts(counts):
    classes = sorted(set().union(*[set(c.keys()) for c in counts.values()]))
    # Header
    print("Class,Train,Val,Test")
    for cls in classes:
        tr = counts["train"].get(cls, 0)
        va = counts["val"].get(cls, 0)
        te = counts["test"].get(cls, 0)
        print(f"{cls},{tr},{va},{te}")
    print(f"Total,{sum(counts['train'].values())},{sum(counts['val'].values())},{sum(counts['test'].values())}")

if __name__ == "__main__":
    dataset_root = "../DATA/OCT2017"  # e.g., ./dataset
    counts = count_split(dataset_root)
    print_counts(counts)
