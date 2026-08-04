import numpy as np
import os
import json


def dirichlet_partition(labels, num_clients=5, alpha=0.5):
    """
    Splits data using a Dirichlet distribution to simulate Non-IID clients.

    Low alpha  (0.1) -> highly skewed, each client dominated by 1-2 grades
    High alpha (1.0) -> mildly skewed, closer to IID
    High alpha (infinite) -> perfectly IID

    Returns a dict mapping client_id -> list of image indices.
    """
    num_classes = len(np.unique(labels))

    # Gather indices per class and shuffle each independently
    class_indices = {c: np.where(labels == c)[0].copy() for c in range(num_classes)}
    client_indices = {i: [] for i in range(num_clients)}

    for c in range(num_classes):
        indices = class_indices[c]
        np.random.shuffle(indices)

        # Sample a probability vector over clients from the Dirichlet distribution
        # np.repeat(alpha, num_clients) -> concentration parameter vector [α, α, α, α, α]
        # Lower α -> more peaked -> one client dominates this class
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

        # Re-normalise — Dirichlet already sums to 1.0 but floating point drift can occur
        proportions = proportions / proportions.sum()

        # Cumulative split: avoids integer truncation loss that happens with per-client astype(int)
        # proportions=[0.4, 0.35, 0.25], len=10 → cumsum=[4, 7, 10] -> slices [0:4],[4:7],[7:10]
        cumulative_splits = np.cumsum(proportions * len(indices)).astype(int)

        # Clamp last value to exact length — prevents off-by-one from floating point rounding
        # cumsum might give [4, 7, 9] instead of [4, 7, 10] for 10 samples -> last gets 1 not 3
        cumulative_splits[-1] = len(indices)

        # Build start positions from cumulative ends: [0, end_0, end_1, ..., end_{n-2}]
        starts = np.concatenate(([0], cumulative_splits[:-1]))

        for i in range(num_clients):
            start = starts[i]
            end   = cumulative_splits[i]
            client_indices[i].extend(indices[start:end].tolist())

    return client_indices


def create_iid_partition(labels, num_clients=5):
    """
    Creates a perfectly equal IID split — every client gets the same
    class distribution and the same number of samples (±1 for rounding).

    Used as the upper-bound reference: FL should approach centralised
    performance under IID conditions.
    """
    # Create a global index array and shuffle it
    indices = np.arange(len(labels))
    np.random.shuffle(indices)

    # array_split handles uneven division automatically
    # 13 samples, 5 clients -> [3, 3, 3, 2, 2] — no samples lost
    splits = np.array_split(indices, num_clients)

    return {i: split.tolist() for i, split in enumerate(splits)}


def geographic_partition(labels, num_clients=5):
    """
    Simulates real-site Non-IID partitioning by assigning fixed clinical
    profiles to each client, mimicking how different hospital types see
    different DR grade distributions in practice.

    Site profiles (grade probabilities [G0, G1, G2, G3, G4]):
        Site 0 — General screening clinic  : [0.70, 0.15, 0.10, 0.03, 0.02]
        Site 1 — Rural primary care        : [0.20, 0.40, 0.25, 0.10, 0.05]
        Site 2 — Specialist referral center: [0.05, 0.10, 0.20, 0.35, 0.30]
        Site 3 — Diabetic eye clinic       : [0.10, 0.15, 0.40, 0.25, 0.10]
        Site 4 — Mixed urban hospital      : [0.20, 0.20, 0.25, 0.20, 0.15]

    Size weights reflect realistic data volume differences between site types:
        Site 0 gets ~30% of total data (highest volume — screening clinics see everyone)
        Site 4 gets ~10% of total data (lowest volume — specialist urban centers)

    Returns a dict mapping client_id -> list of image indices.
    """
    assert num_clients == 5, (
        f"Geographic partition is designed for exactly 5 sites, got {num_clients}."
    )

    # Each row = P(grade | site). Rows sum to 1.0.
    # These profiles are clinically motivated:
    #   - Screening clinics see mostly healthy patients (G0 dominant)
    #   - Specialist referral centers see the worst cases (G3/G4 dominant)
    #   - Rural primary care misses severe cases (G1/G2 dominant)
    site_profiles = np.array([
        [0.70, 0.15, 0.10, 0.03, 0.02],   # Site 0: general screening clinic
        [0.20, 0.40, 0.25, 0.10, 0.05],   # Site 1: rural primary care
        [0.05, 0.10, 0.20, 0.35, 0.30],   # Site 2: specialist referral center
        [0.10, 0.15, 0.40, 0.25, 0.10],   # Site 3: diabetic eye clinic
        [0.20, 0.20, 0.25, 0.20, 0.15],   # Site 4: mixed urban hospital
    ])

    # Proportion of total dataset each site receives
    # Large screening clinic (30%) vs small specialist center (10%)
    site_size_weights = np.array([0.30, 0.25, 0.15, 0.20, 0.10])
    site_size_weights = site_size_weights / site_size_weights.sum()  # normalise to sum=1

    num_classes  = 5
    num_samples  = len(labels)

    # Gather and shuffle indices per class
    class_indices = {c: np.where(labels == c)[0].copy() for c in range(num_classes)}
    for c in range(num_classes):
        np.random.shuffle(class_indices[c])

    client_indices = {i: [] for i in range(num_clients)}

    for c in range(num_classes):
        indices           = class_indices[c]
        total_this_class  = len(indices)

        if total_this_class == 0:
            # Skip grades that don't appear in the dataset (shouldn't happen after class-5 filter)
            continue

        # Joint demand: how much of class c does each site want?
        # raw_demand[site] = P(site) * P(grade=c | site)
        # This gives the unnormalised "pull" each site has on this grade's samples
        raw_demand = site_size_weights * site_profiles[:, c]

        if raw_demand.sum() == 0:
            # Fallback: if no site has any profile weight for this grade, split evenly
            raw_demand = np.ones(num_clients)

        # Normalise across sites so all samples of this class are assigned
        proportions = raw_demand / raw_demand.sum()

        # Same cumulative split logic as dirichlet_partition — no truncation loss
        cumulative_splits = np.cumsum(proportions * total_this_class).astype(int)
        cumulative_splits[-1] = total_this_class   # clamp last to exact count

        starts = np.concatenate(([0], cumulative_splits[:-1]))

        for i in range(num_clients):
            start = starts[i]
            end   = cumulative_splits[i]
            client_indices[i].extend(indices[start:end].tolist())

    return client_indices


def load_partition(partition_path):
    """
    Loads a saved partition JSON and casts string keys back to int.

    JSON always serialises dict keys as strings ("0", "1", ...),
    so mapping[0] would raise KeyError without this conversion.

    Usage:
        mapping = load_partition("data/partitions/iid.json")
        client_0_indices = mapping[0]   # works correctly
    """
    with open(partition_path, 'r') as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


if __name__ == "__main__":

    #1. Load labels from DDR train.txt
    data_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'data', 'DDR', 'DR_grading')
    )
    txt_path = os.path.join(data_dir, "train.txt")

    labels = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # Each line: "<image_name> <label>"
            # Skip malformed lines and ungradable images (Grade 5)
            if len(parts) == 2 and int(parts[1]) != 5:
                labels.append(int(parts[1]))

    labels = np.array(labels)
    print(f"Loaded {len(labels)} valid training labels.")

    # Quick sanity check — all 5 grades must be present
    unique_grades, grade_counts = np.unique(labels, return_counts=True)
    print("Grade distribution in full dataset:")
    for grade, count in zip(unique_grades, grade_counts):
        print(f"  Grade {grade}: {count} ({count/len(labels)*100:.1f}%)")

    # 2. Create output directory
    output_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'data', 'partitions')
    )
    os.makedirs(output_dir, exist_ok=True)

    #3. Define all partition scenarios
    scenarios = {
        "real_site":      lambda: geographic_partition(labels),
        "dirichlet_0.1":  lambda: dirichlet_partition(labels, alpha=0.1),
        "dirichlet_0.5":  lambda: dirichlet_partition(labels, alpha=0.5),
        "dirichlet_1.0":  lambda: dirichlet_partition(labels, alpha=1.0),
        "iid":            lambda: create_iid_partition(labels),
    }

    #4. Generate, validate, and save each partition
    for name, func in scenarios.items():
        print(f"\n{'='*50}")
        print(f"Generating '{name}' partition...")

        mapping = func()

        #Integrity check: no sample lost or double-counted
        total_assigned = sum(len(v) for v in mapping.values())
        assert total_assigned == len(labels), (
            f"[{name}] Sample count mismatch! "
            f"Assigned {total_assigned} but expected {len(labels)}"
        )

        # Check no index appears in two clients
        all_assigned = [idx for v in mapping.values() for idx in v]
        assert len(all_assigned) == len(set(all_assigned)), (
            f"[{name}] Duplicate indices found — some samples assigned to multiple clients!"
        )

        #Save to JSON
        out_path = os.path.join(output_dir, f"{name}.json")
        with open(out_path, 'w') as f:
            json.dump(mapping, f, indent=4)

        #Print per-client grade distribution
        for i in range(5):
            client_labels          = labels[mapping[i]]
            unique, counts         = np.unique(client_labels, return_counts=True)
            dist                   = dict(zip(unique.tolist(), counts.tolist()))
            total                  = len(client_labels)

            # Format as percentages for easy skew inspection
            dist_pct = {
                f"G{g}": f"{counts[j]/total*100:.0f}%"
                for j, g in enumerate(unique)
            }
            print(f"  Client {i}: {total:>5} imgs | {dist_pct}")

    print(f"\n✅ All 5 partitions saved to: {output_dir}")
    print("   Files: real_site.json, dirichlet_0.1.json, dirichlet_0.5.json, "
          "dirichlet_1.0.json, iid.json")