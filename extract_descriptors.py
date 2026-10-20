import os, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

NPZ = "data/processed_data.npz"
OUT = "data/rdkit_descriptors.npz"

DESC_NAMES = ["MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumAromaticRings",
              "NumRotatableBonds", "MolMR", "FractionCSP3", "NumHeavyAtoms", "RingCount",
              "NumSaturatedRings", "NumAliphaticRings", "BertzCT", "LabuteASA",
              "BalabanJ", "Chi0", "Chi1", "HallKierAlpha", "Kappa1", "NumValenceElectrons"]


def calc_one(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return np.zeros(len(DESC_NAMES), dtype=np.float32)
    return np.array([
        Crippen.MolLogP(m), Descriptors.TPSA(m), Lipinski.NumHDonors(m),
        Lipinski.NumHAcceptors(m), Descriptors.NumAromaticRings(m),
        Descriptors.NumRotatableBonds(m), Crippen.MolMR(m), Descriptors.FractionCSP3(m),
        float(m.GetNumHeavyAtoms()), Descriptors.RingCount(m),
        Descriptors.NumSaturatedRings(m), Descriptors.NumAliphaticRings(m),
        Descriptors.BertzCT(m), Descriptors.LabuteASA(m),
        Descriptors.BalabanJ(m), Descriptors.Chi0(m), Descriptors.Chi1(m),
        Descriptors.HallKierAlpha(m), Descriptors.Kappa1(m),
        float(Descriptors.NumValenceElectrons(m)),
    ], dtype=np.float32)


def main():
    d = np.load(NPZ, allow_pickle=True)


    df = pd.read_excel("data/used_dataset.xlsx")
    assert len(df) == len(d["y"]), f"行数不符: xlsx {len(df)} vs npz {len(d['y'])}"
    smi_all = df["SMILES"].astype(str).values


    uniq = list(dict.fromkeys(smi_all))
    print(f"  唯一 SMILES: {len(uniq)}  总行数: {len(smi_all)}")
    cache = {}
    for i, s in enumerate(uniq):
        cache[s] = calc_one(s)
        if (i + 1) % 1000 == 0:
            print(f"    {i+1}/{len(uniq)}")

    desc = np.stack([cache[s] for s in smi_all]).astype(np.float32)
    print(f"  descriptors shape: {desc.shape}")

    from sklearn.preprocessing import StandardScaler
    desc_norm = StandardScaler().fit_transform(desc).astype(np.float32)
    np.savez(OUT, descriptors=desc, descriptors_norm=desc_norm, names=np.array(DESC_NAMES))
    print(f"  saved -> {OUT}")
    print("  描述符:", DESC_NAMES)
    print("  均值(标准化后应~0):", desc_norm.mean(0).round(3))
    print("  标准差(标准化后应~1):", desc_norm.std(0).round(3))


if __name__ == "__main__":
    main()
