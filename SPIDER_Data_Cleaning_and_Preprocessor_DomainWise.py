from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# =========================================================
# Config
# =========================================================
SRC_ROOT = Path("SPIDER_normalized")
OUT_ROOT = Path("SPIDER_domain_strict")

IMAGES_DIR = SRC_ROOT / "images"
MASKS_DIR = SRC_ROOT / "masks"
OVERVIEW_CSV = SRC_ROOT / "overview.csv"
FINAL_SUMMARY_CSV = SRC_ROOT / "logs" / "final_summary.csv"

# 严格实验池条件
MIN_PATIENTS_PER_DOMAIN = 10
KEEP_PROTOCOLS = {"T1-TSE", "T2-TSE"}

# 复制模式：
#   "copy"    -> 真复制文件
#   "symlink" -> 建符号链接（省空间，Linux/macOS 更方便）
COPY_MODE = "copy"


# =========================================================
# Helpers
# =========================================================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def normalize_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_serial(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return ""
    # 处理 70714.0 这种情况
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def slugify(s: str) -> str:
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def parse_case_id_from_stem(stem: str) -> str:
    """
    输入:
        1_t1
        12_t2
        104_t1
    输出:
        1 / 12 / 104
    """
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        return stem
    return parts[0]


def classify_protocol(row: pd.Series) -> str:
    """
    严格 protocol 判定：
    1) 先用 SeriesDescription / SequenceName / ScanningSequence 识别 SPACE / STIR
    2) 其余按 modality 落到 T1-TSE / T2-TSE
    """
    modality = normalize_text(row.get("modality", "")).lower()
    text = " ".join([
        normalize_text(row.get("SeriesDescription", "")),
        normalize_text(row.get("SequenceName", "")),
        normalize_text(row.get("ScanningSequence", "")),
    ]).lower()

    # 先排除会混淆 strict pool 的协议
    if ("space" in text) or ("spc" in text):
        return "T2-SPACE"
    if ("stir" in text) or ("tir" in text):
        return "T2-STIR"

    # 主线严格协议
    if modality == "t1":
        return "T1-TSE"
    if modality == "t2":
        return "T2-TSE"

    return "OTHER"


def safe_copy(src: Path, dst: Path, mode: str = "copy") -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


# =========================================================
# Main
# =========================================================
def main():
    if not OVERVIEW_CSV.exists():
        raise FileNotFoundError(f"Missing: {OVERVIEW_CSV}")
    if not FINAL_SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing: {FINAL_SUMMARY_CSV}")
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Missing: {IMAGES_DIR}")
    if not MASKS_DIR.exists():
        raise FileNotFoundError(f"Missing: {MASKS_DIR}")

    ensure_dir(OUT_ROOT)

    # -----------------------------------------------------
    # 1) 读取清洗后保留样本 + 原始元数据
    # -----------------------------------------------------
    final_df = pd.read_csv(FINAL_SUMMARY_CSV)
    overview_df = pd.read_csv(OVERVIEW_CSV)

    # 只保留最终真正 saved 的样本
    final_df = final_df.copy()
    final_df["kept"] = final_df["kept"].astype(bool)
    final_df = final_df[final_df["kept"]].copy()

    # 文件 stem，如 100_t1
    final_df["stem"] = final_df["file"].astype(str).str.replace(".mha", "", regex=False)

    # overview 中 new_file_name 正好对应 100_t1 / 100_t2
    overview_df = overview_df.copy()
    overview_df["new_file_name"] = overview_df["new_file_name"].astype(str)

    # 合并
    df = final_df.merge(
        overview_df,
        left_on="stem",
        right_on="new_file_name",
        how="left",
        validate="one_to_one",
    )

    # -----------------------------------------------------
    # 2) 构造 patient_id / center_id / protocol / domain_id
    # -----------------------------------------------------
    df["patient_id"] = df["stem"].apply(parse_case_id_from_stem)

    df["Manufacturer_norm"] = df["Manufacturer"].apply(normalize_text)
    df["ManufacturerModelName_norm"] = df["ManufacturerModelName"].apply(normalize_text)
    df["DeviceSerialNumber_norm"] = df["DeviceSerialNumber"].apply(normalize_serial)

    df["protocol"] = df.apply(classify_protocol, axis=1)

    df["center_id_raw"] = (
        df["Manufacturer_norm"] + "__" +
        df["ManufacturerModelName_norm"] + "__" +
        df["DeviceSerialNumber_norm"]
    )

    df["center_id"] = df["center_id_raw"].apply(slugify)
    df["domain_id_raw"] = df["center_id_raw"] + "__" + df["protocol"]
    df["domain_id"] = df["domain_id_raw"].apply(slugify)

    # 标记文件路径
    df["image_path"] = df["file"].apply(lambda x: str((IMAGES_DIR / x).resolve()))
    df["mask_path"] = df["file"].apply(lambda x: str((MASKS_DIR / x).resolve()))

    # -----------------------------------------------------
    # 3) 严格过滤：serial 非空 + protocol 合法
    # -----------------------------------------------------
    strict_df = df[
        (df["DeviceSerialNumber_norm"] != "") &
        (df["protocol"].isin(KEEP_PROTOCOLS))
    ].copy()

    # -----------------------------------------------------
    # 4) 统计 domain 内患者数，只保留 >= MIN_PATIENTS_PER_DOMAIN
    # -----------------------------------------------------
    group_stats = (
        strict_df.groupby(
            ["domain_id", "center_id", "protocol",
             "Manufacturer_norm", "ManufacturerModelName_norm", "DeviceSerialNumber_norm"],
            dropna=False
        )
        .agg(
            n_files=("file", "count"),
            n_patients=("patient_id", "nunique"),
            modalities=("modality", lambda x: ",".join(sorted(set(map(str, x))))),
        )
        .reset_index()
        .sort_values(["protocol", "n_patients", "domain_id"], ascending=[True, False, True])
    )

    keep_domains = set(
        group_stats.loc[group_stats["n_patients"] >= MIN_PATIENTS_PER_DOMAIN, "domain_id"].tolist()
    )

    kept_df = strict_df[strict_df["domain_id"].isin(keep_domains)].copy()

    kept_group_stats = group_stats[group_stats["domain_id"].isin(keep_domains)].copy()

    # -----------------------------------------------------
    # 5) 导出统计表
    # -----------------------------------------------------
    df.to_csv(OUT_ROOT / "all_merged_samples.csv", index=False)
    strict_df.to_csv(OUT_ROOT / "strict_candidate_samples.csv", index=False)
    group_stats.to_csv(OUT_ROOT / "strict_candidate_group_stats.csv", index=False)
    kept_df.to_csv(OUT_ROOT / "strict_kept_samples.csv", index=False)
    kept_group_stats.to_csv(OUT_ROOT / "strict_kept_group_stats.csv", index=False)

    # 每个 domain 的患者列表
    patient_manifest_rows = []
    for domain_id, sub in kept_df.groupby("domain_id"):
        patient_ids = sorted(sub["patient_id"].astype(str).unique().tolist())
        for pid in patient_ids:
            patient_manifest_rows.append({
                "domain_id": domain_id,
                "patient_id": pid,
            })
    pd.DataFrame(patient_manifest_rows).to_csv(
        OUT_ROOT / "strict_domain_patient_manifest.csv", index=False
    )

    # -----------------------------------------------------
    # 6) 按 domain 建目录并复制/链接 image + mask
    # -----------------------------------------------------
    for domain_id, sub in kept_df.groupby("domain_id"):
        domain_root = OUT_ROOT / domain_id
        domain_img_dir = domain_root / "images"
        domain_msk_dir = domain_root / "masks"
        ensure_dir(domain_img_dir)
        ensure_dir(domain_msk_dir)

        # 保存该 domain 的 manifest
        sub_sorted = sub.sort_values(["patient_id", "modality", "file"]).copy()
        sub_sorted.to_csv(domain_root / "manifest.csv", index=False)

        for _, row in sub_sorted.iterrows():
            img_src = Path(row["image_path"])
            msk_src = Path(row["mask_path"])

            if not img_src.exists():
                raise FileNotFoundError(f"Image not found: {img_src}")
            if not msk_src.exists():
                raise FileNotFoundError(f"Mask not found: {msk_src}")

            img_dst = domain_img_dir / img_src.name
            msk_dst = domain_msk_dir / msk_src.name

            safe_copy(img_src, img_dst, mode=COPY_MODE)
            safe_copy(msk_src, msk_dst, mode=COPY_MODE)

    # -----------------------------------------------------
    # 7) 打印摘要
    # -----------------------------------------------------
    print("\n===== Strict domain grouping done =====")
    print(f"Source root : {SRC_ROOT.resolve()}")
    print(f"Output root : {OUT_ROOT.resolve()}")
    print(f"Copy mode   : {COPY_MODE}")
    print(f"Min patients/domain: {MIN_PATIENTS_PER_DOMAIN}")
    print("\nKept strict domains:")

    if len(kept_group_stats) == 0:
        print("No strict domain satisfies the current filter.")
    else:
        for _, r in kept_group_stats.iterrows():
            print(
                f"  - {r['domain_id']}: "
                f"{r['n_patients']} patients, {r['n_files']} files"
            )

    print("\nSaved files:")
    print(f"  - {OUT_ROOT / 'all_merged_samples.csv'}")
    print(f"  - {OUT_ROOT / 'strict_candidate_samples.csv'}")
    print(f"  - {OUT_ROOT / 'strict_candidate_group_stats.csv'}")
    print(f"  - {OUT_ROOT / 'strict_kept_samples.csv'}")
    print(f"  - {OUT_ROOT / 'strict_kept_group_stats.csv'}")
    print(f"  - {OUT_ROOT / 'strict_domain_patient_manifest.csv'}")


if __name__ == "__main__":
    main()