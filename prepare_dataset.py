import os, sys, zipfile, yaml, re, warnings
import numpy as np, pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect as MorganFP
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
from rdkit.Chem.Draw import rdMolDraw2D
warnings.filterwarnings('ignore')

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def il_smiles_to_image(smiles, img_size=224, save_path=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try:
        AllChem.Compute2DCoords(mol)
    except: pass
    drawer = rdMolDraw2D.MolDraw2DCairo(img_size, img_size)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    img_data = drawer.GetDrawingText()
    if save_path and img_data:
        with open(save_path, "wb") as f: f.write(img_data)
    return img_data

def generate_random_smiles(smiles, n_variants=3):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]
    result = [smiles]
    try:
        rand_smis = Chem.MolToRandomSmilesVect(mol, n_variants, randomSeed=42)
        result.extend(rand_smis)
    except:
        pass
    return result


def compute_fingerprints(smiles):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(2214, dtype=np.float32)
    maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)[1:]
    ecfp = MorganFP(mol, 2, nBits=2048)
    ecfp_arr = np.array(ecfp, dtype=np.float32)
    return np.concatenate([maccs, ecfp_arr])

def parse_smiles_anion(smiles):

    parts = str(smiles).split('.')
    return parts[-1] if len(parts) > 1 else parts[0]

def main():
    config = load_config()
    cfg_data = config["data"]
    os.makedirs(cfg_data["molecular_images_dir"], exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("="*60)
    print("多模态离子液体毒性预测 — 数据准备")
    print("="*60)


    print("\n[1/7] 读取CSV数据...")
    z = zipfile.ZipFile(cfg_data["dataset_zip"], 'r')
    all_rows = []
    for name in z.namelist():
        if name in ['cell lines.csv', 'methods.csv', 'media.csv']:
            continue
        with z.open(name) as f:
            try:
                df = pd.read_csv(f)
                all_rows.append(df)
            except: pass


    merged = pd.concat(all_rows, ignore_index=True, join='outer')
    z.close()
    print(f"总记录: {len(merged)}")


    print("\n[2/7] 统一列名与清洗...")


    tox_cols = [c for c in merged.columns if ('EC50' in str(c) or 'CC50' in str(c) or 'IC50' in str(c)) and 'Statistics' not in str(c) and 'statistics' not in str(c)]
    merged['Tox_raw'] = merged[tox_cols[0]] if tox_cols else np.nan
    for tc in tox_cols[1:]:
        merged['Tox_raw'] = merged['Tox_raw'].fillna(merged[tc])

    for tc in tox_cols:
        if tc != 'Tox_raw' and tc in merged.columns:
            merged.drop(columns=[tc], inplace=True)


    mw_cols = [c for c in merged.columns if ('Mw' in str(c) or 'mw' in str(c))]
    merged['Mw'] = merged[mw_cols[0]] if mw_cols else 0
    for mc in mw_cols[1:]:
        merged['Mw'] = merged['Mw'].fillna(merged[mc])
    for mc in mw_cols:
        if mc != 'Mw' and mc in merged.columns:
            merged.drop(columns=[mc], inplace=True)
    merged['Mw'] = pd.to_numeric(merged['Mw'], errors='coerce')


    print("\n[3/7] 清洗毒性值...")
    print(f"  SMILES缺失: {merged['SMILES'].isna().sum() if 'SMILES' in merged.columns else 'N/A'}")


    merged['Tox_str'] = merged['Tox_raw'].astype(str).str.strip()
    merged['Tox_str'] = merged['Tox_str'].str.replace('[<>~,;]', '', regex=True)
    merged['Tox_float'] = pd.to_numeric(merged['Tox_str'], errors='coerce')


    valid = merged['Tox_float'].notna() & (merged['Tox_float'] > 0)
    valid &= merged['SMILES'].notna()
    valid &= merged['Mw'].notna() & (merged['Mw'] > 0)
    df_clean = merged[valid].copy()


    df_clean['logEC50'] = np.log10(df_clean['Tox_float'] * 1000)


    df_clean['Anion_SMILES'] = df_clean['SMILES'].apply(parse_smiles_anion)

    print(f"  有效记录: {len(df_clean)}")
    print(f"  logEC50范围: [{df_clean['logEC50'].min():.2f}, {df_clean['logEC50'].max():.2f}]")


    print("\n[4/7] 编码细胞系...")
    cl_name = None
    for c in df_clean.columns:
        if 'Cell' in c or 'cell' in c:
            cl_name = c; break
    if cl_name: df_clean.rename(columns={cl_name: 'Cell_line'}, inplace=True)

    top_n = 10
    top_cells = df_clean['Cell_line'].value_counts().head(top_n).index.tolist()
    df_clean['Cell_encoded'] = df_clean['Cell_line'].apply(
        lambda x: x if x in top_cells else 'Other'
    )

    df_clean = df_clean[~df_clean['Cell_encoded'].isin(['HCT-116', 'T-47D'])]
    print(f"  细胞系类别: {df_clean['Cell_encoded'].nunique()} (top {top_n} + Other)")
    print(f"  Top: {df_clean['Cell_encoded'].value_counts().head(5).to_dict()}")


    print("\n[4.5/7] 生成随机 SMILES 变体 (每个分子 2 种)...")
    aug_rows = []
    for idx, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="  增强"):
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is None:
            continue
        try:
            rand_smis = Chem.MolToRandomSmilesVect(mol, 2, randomSeed=int(idx) % 1000)
            for rs in rand_smis:
                new_row = row.copy()
                new_row["SMILES"] = rs
                aug_rows.append(new_row)
        except:
            pass
    if aug_rows:
        df_aug = pd.DataFrame(aug_rows)
        df_clean = pd.concat([df_clean, df_aug], ignore_index=True)
    print(f"  增强后总记录: {len(df_clean)} (原始 {len(df_clean) - len(aug_rows)} + 增强 {len(aug_rows)})")


    print("\n[5/7] 计算MACCS+ECFP指纹(2214维)...")
    fps_list = []
    failed_fp = 0
    for _, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="  指纹"):
        fp = compute_fingerprints(row['SMILES'])
        if fp.sum() == 0: failed_fp += 1
        fps_list.append(fp)
    X_num = np.array(fps_list, dtype=np.float32)
    print(f"  指纹形状: {X_num.shape}, 失败: {failed_fp}")


    print("\n[6/7] 生成2D分子结构图...")
    img_dir = cfg_data["molecular_images_dir"]
    img_paths = []
    failed_img = 0
    for i, (_, row) in tqdm(enumerate(df_clean.iterrows()), total=len(df_clean), desc="  绘图"):
        img_name = f"il_{i:05d}.png"
        img_path = os.path.join(img_dir, img_name)
        result = il_smiles_to_image(row['SMILES'], save_path=img_path)
        if result is None: failed_img += 1
        img_paths.append(img_path)
    print(f"  成功: {len(img_paths)-failed_img}, 失败: {failed_img}")


    print("\n[7/7] 划分训练/验证/测试集...")
    from sklearn.model_selection import train_test_split
    groups = df_clean['Cell_encoded'].values
    indices = np.arange(len(df_clean))


    from sklearn.model_selection import train_test_split as tts
    train_val_idx, test_idx = tts(indices, test_size=0.1, random_state=42)
    train_idx, val_idx = tts(train_val_idx, test_size=0.1/0.9, random_state=42)

    print(f"  训练: {len(train_idx)}, 验证: {len(val_idx)}, 测试: {len(test_idx)}")


    y = df_clean['logEC50'].values.astype(np.float32)
    Mw = df_clean['Mw'].values.astype(np.float32)

    from sklearn.preprocessing import OneHotEncoder
    oh = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cell_oh = oh.fit_transform(df_clean[['Cell_encoded']])

    np.savez(cfg_data['processed_data_path'],
             X_num=X_num, y=y, Mw=Mw, cell_oh=cell_oh,
             img_paths=np.array(img_paths),
             train_idx=np.array(train_idx), val_idx=np.array(val_idx),
             test_idx=np.array(test_idx),
             cell_labels=df_clean['Cell_encoded'].values, cell_categories=list(oh.categories_[0]))


    print("\n生成使用数据集Excel...")
    excel_out = pd.DataFrame({
        'IL编号': range(1, len(df_clean)+1),
        'IL名称': df_clean['Name'].values,
        'SMILES': df_clean['SMILES'].values,
        '阴离子SMILES': df_clean['Anion_SMILES'].values,
        '细胞系': df_clean['Cell_encoded'].values,
        '测试方法': df_clean.get('Method', 'N/A').values,
        '毒性值(mM)': df_clean['Tox_float'].values,
        '毒性值(logEC50)': df_clean['logEC50'].values,
        '分子量(Mw)': df_clean['Mw'].values,
        '数据集划分': [''] * len(df_clean),
    })
    excel_out.loc[train_idx, '数据集划分'] = '训练'
    excel_out.loc[val_idx, '数据集划分'] = '验证'
    excel_out.loc[test_idx, '数据集划分'] = '测试'
    excel_out.to_excel(cfg_data['used_dataset_path'], index=False)
    print(f"  Excel已保存: {cfg_data['used_dataset_path']}")

    print("\n✓ 数据准备完成！")

if __name__ == '__main__':
    main()
