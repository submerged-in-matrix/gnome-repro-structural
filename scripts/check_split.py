"""Sanity-check the dataset and the composition-based split."""

from __future__ import annotations

from collections import Counter

from gnome.data import load_mp_data, assign_split


def main():
    print("Loading MP entries from local cache...")
    entries = load_mp_data(verbose=True)
    print(f"\nLoaded {len(entries)} entries.")

    # 1. Split assignment
    splits = Counter(assign_split(e["formula_pretty"]) for e in entries)
    n_total = sum(splits.values())
    train_frac = splits["train"] / n_total
    print(f"\nSplit sizes:")
    print(f"  train: {splits['train']:>7d}  ({100*train_frac:.1f}%)")
    print(f"  test:  {splits['test']:>7d}  ({100*(1-train_frac):.1f}%)")

    # 2. Composition-overlap check (the key invariant)
    train_compositions = {e["formula_pretty"] for e in entries
                          if assign_split(e["formula_pretty"]) == "train"}
    test_compositions = {e["formula_pretty"] for e in entries
                         if assign_split(e["formula_pretty"]) == "test"}
    overlap = train_compositions & test_compositions
    print(f"\nUnique compositions: train={len(train_compositions)}, "
          f"test={len(test_compositions)}")
    print(f"Composition overlap: {len(overlap)}  (must be 0)")
    assert len(overlap) == 0, "BUG: split has composition overlap"

    # 3. Element coverage
    def elements_in(split_name):
        elts = Counter()
        for e in entries:
            if assign_split(e["formula_pretty"]) != split_name:
                continue
            for site in e["structure"]:
                elts[site.specie.symbol] += 1
        return elts

    train_elts = elements_in("train")
    test_elts = elements_in("test")
    train_only = set(train_elts) - set(test_elts)
    test_only = set(test_elts) - set(train_elts)
    print(f"\nElements: train has {len(train_elts)}, test has {len(test_elts)}")
    if train_only:
        print(f"  in train only: {sorted(train_only)}")
    if test_only:
        print(f"  in test only:  {sorted(test_only)}")

    # 4. Formation-energy distributions
    import statistics
    train_e = [e["e_form_per_atom"] for e in entries
               if assign_split(e["formula_pretty"]) == "train"]
    test_e = [e["e_form_per_atom"] for e in entries
              if assign_split(e["formula_pretty"]) == "test"]
    print(f"\nFormation energy (eV/atom):")
    print(f"  train: mean={statistics.mean(train_e):+.3f}  "
          f"std={statistics.stdev(train_e):.3f}  "
          f"min={min(train_e):+.3f}  max={max(train_e):+.3f}")
    print(f"  test:  mean={statistics.mean(test_e):+.3f}  "
          f"std={statistics.stdev(test_e):.3f}  "
          f"min={min(test_e):+.3f}  max={max(test_e):+.3f}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()