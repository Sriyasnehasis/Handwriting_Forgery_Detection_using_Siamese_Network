import os
import random
import itertools
import pickle

genuine_dir = 'data/raw/cedardataset/genuine'
forged_dir = 'data/raw/cedardataset/forged'

def get_user_from_genuine_filename(filename):
    return filename.split('_')[1]

def get_user_from_forged_filename(filename):
    return filename.split('_')[1]

genuine_files = [f for f in os.listdir(genuine_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
forged_files = [f for f in os.listdir(forged_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

genuine_by_user = {}
for f in genuine_files:
    uid = get_user_from_genuine_filename(f)
    genuine_by_user.setdefault(uid, []).append(f)

forged_by_user = {}
for f in forged_files:
    uid = get_user_from_forged_filename(f)
    forged_by_user.setdefault(uid, []).append(f)

pairs = []
labels = []

for user, genuines in genuine_by_user.items():
    # Positive pairs
    for (f1, f2) in itertools.combinations(genuines, 2):
        pairs.append((
            os.path.join(genuine_dir, f1),
            os.path.join(genuine_dir, f2)
        ))
        labels.append(1)
    # Negative pairs:
    forgeds = forged_by_user.get(user, [])
    for f_genuine in genuines:
        for f_forged in forgeds:
            pairs.append((
                os.path.join(genuine_dir, f_genuine),
                os.path.join(forged_dir, f_forged)
            ))
            labels.append(0)

# Optionally, cross-user negative pairs:
user_ids = list(genuine_by_user.keys())
for i, userA in enumerate(user_ids):
    for userB in user_ids[i+1:]:
        gA = random.choice(genuine_by_user[userA])
        gB = random.choice(genuine_by_user[userB])
        pairs.append((
            os.path.join(genuine_dir, gA),
            os.path.join(genuine_dir, gB)
        ))
        labels.append(0)

print(f"Total pairs: {len(pairs)} (Positives: {sum(labels)}, Negatives: {len(labels)-sum(labels)})")

with open('data/processed/sig_pairs_labels.pkl', 'wb') as f:
    pickle.dump({'pairs': pairs, 'labels': labels}, f)

print("✅ Pairs and labels saved to data/processed/sig_pairs_labels.pkl")
