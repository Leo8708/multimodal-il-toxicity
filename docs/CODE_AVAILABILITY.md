# Manuscript Code availability wording

The text below is a template. Replace the repository placeholder only after creating the actual repository, uploading the files and checking public access. No repository URL or archive DOI has been assigned in this package.

## After the GitHub repository is publicly accessible

> The core implementations for data preprocessing, molecular feature extraction, and FNN, DCNN and XGBoost modeling are available at [GITHUB_REPOSITORY_URL] under the MIT License. The repository README describes the included workflows, input requirements and reproducibility limitations.

This wording accurately describes a foundational code release. Do not replace it with “all code required to reproduce all results” unless that stronger statement has been verified. The later grouped scripts are not an exact reconstruction of the manuscript's historical grouped table.

## After archiving a specific release, if an archive is created

> The core implementations for data preprocessing, molecular feature extraction, and FNN, DCNN and XGBoost modeling are available at [GITHUB_REPOSITORY_URL] under the MIT License. Version [RELEASE_VERSION] is archived at [SOFTWARE_ARCHIVE_DOI]. The repository README describes the included workflows, input requirements and reproducibility limitations.

The version/DOI wording applies only after an actual release and archive record exist. A GitHub release can provide a fixed code version; an optional archive such as Zenodo can provide a persistent software DOI. This package has not created either.

## 中文核对

- 仓库公开并核实链接可访问后，才能在论文中使用 “are available at”。
- 将 `[GITHUB_REPOSITORY_URL]` 替换为真实地址；仅在确实归档后填写版本号和软件 DOI。
- 此声明只描述代码。数据来源、确切文件的获取地址及数据许可需要在 Data Availability 或数据说明中另行交代。
- 当前代码包没有承诺逐项复现稿件全部数值；历史 grouped 训练版本尚未找回，不应扩大声明范围。
- 上述为通用英文模板；目标期刊如有专门要求，还需按该期刊格式调整。
