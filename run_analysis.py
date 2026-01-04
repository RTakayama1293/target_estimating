#!/usr/bin/env python3
"""
EEZO BtoC ターゲット推定分析 - メイン実行スクリプト
CLAUDE.mdに従った分析フロー全体を実行
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 日本語フォント設定
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False


def run_experiment(exp_dir: Path) -> bool:
    """実験スクリプトを実行する。

    Args:
        exp_dir: 実験ディレクトリ

    Returns:
        成功した場合True
    """
    script_path = exp_dir / 'analyze.py'
    if not script_path.exists():
        logger.error(f"スクリプトが見つかりません: {script_path}")
        return False

    logger.info(f"実行中: {exp_dir.name}")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        logger.error(f"エラー: {result.stderr}")
        return False

    logger.info(result.stdout)
    return True


def generate_final_report(project_dir: Path) -> None:
    """最終統合レポートを生成する。

    Args:
        project_dir: プロジェクトルートディレクトリ
    """
    output_dir = project_dir / 'outputs'
    exp_base = project_dir / 'experiments'

    # 各実験の結果を読み込む
    baseline_summary = (exp_base / 'exp001_baseline' / 'outputs' / 'base_case_summary.md').read_text(encoding='utf-8')
    sensitivity_report = (exp_base / 'exp002_sensitivity' / 'outputs' / 'key_drivers.md').read_text(encoding='utf-8')
    recommendation = (exp_base / 'exp003_scenarios' / 'outputs' / 'recommendation.md').read_text(encoding='utf-8')

    # シナリオ比較データを読み込む
    scenario_df = pd.read_csv(exp_base / 'exp003_scenarios' / 'outputs' / 'scenario_comparison.csv')

    # 統合レポート生成
    final_report = f"""# EEZO BtoC ターゲット推定分析 - 最終レポート

**作成日**: {datetime.now().strftime('%Y年%m月%d日')}

---

## エグゼクティブサマリー

### 分析の目的
フェリーユーザー（年間36万人）をベースとしたEEZO ECサイトのターゲット設定と商品戦略の最適化

### 主要な発見

1. **現状（ベースケース）での目標達成は困難**
   - FY2026目標: 45百万円
   - 現状予測: ベースケースで{scenario_df[scenario_df['シナリオ']=='ベースケース']['売上'].values[0]/1e6:.1f}百万円

2. **最も効果的な施策**
   - S2・S3（富裕層・準富裕層）への到達率向上が最大インパクト
   - 市場規模の大きいセグメントへのアプローチが効率的

3. **推奨戦略**
   - **シナリオB（ボリューム狙い）をベースに、シナリオC（高単価化）を組み合わせ**
   - 短期: 到達率向上 → 中期: 価格帯シフト → 継続: S1向けVIP施策

### 数値サマリー

| 項目 | ベースケース | シナリオB（推奨） | 変化率 |
|-----|------------|-----------------|--------|
| 売上 | {scenario_df[scenario_df['シナリオ']=='ベースケース']['売上'].values[0]/1e6:.2f}M | {scenario_df[scenario_df['シナリオ']=='シナリオB']['売上'].values[0]/1e6:.2f}M | +{scenario_df[scenario_df['シナリオ']=='シナリオB']['売上変化率'].values[0]:.1f}% |
| 粗利 | {scenario_df[scenario_df['シナリオ']=='ベースケース']['粗利'].values[0]/1e6:.2f}M | {scenario_df[scenario_df['シナリオ']=='シナリオB']['粗利'].values[0]/1e6:.2f}M | +{scenario_df[scenario_df['シナリオ']=='シナリオB']['粗利変化率'].values[0]:.1f}% |
| LTV | {scenario_df[scenario_df['シナリオ']=='ベースケース']['LTV'].values[0]/1e6:.2f}M | {scenario_df[scenario_df['シナリオ']=='シナリオB']['LTV'].values[0]/1e6:.2f}M | +{scenario_df[scenario_df['シナリオ']=='シナリオB']['LTV変化率'].values[0]:.1f}% |

---

## 目次

1. [ベースケース分析](#ベースケース分析)
2. [感度分析](#感度分析)
3. [シナリオ比較と戦略推奨](#シナリオ比較と戦略推奨)
4. [次のステップ](#次のステップ)

---

## ベースケース分析

{baseline_summary}

---

## 感度分析

{sensitivity_report}

---

## シナリオ比較と戦略推奨

{recommendation}

---

## 次のステップ

### 優先度1: データ精度の向上
- 現在の転換率（0.04-0.09%）と想定転換率（3-8%）のギャップ検証
- 実際のセグメント構成比の確認

### 優先度2: 施策の具体化
- フェリー船内でのEC認知施策の具体案作成
- メルマガセグメント配信の設計

### 優先度3: KPI設定
- 到達率・転換率の月次モニタリング体制
- Shopify移行後のデータ取得設計

---

## 付録

### 分析に使用したパラメータ
- パラメータファイル: `data/parameters.json`
- 各実験の詳細結果: `experiments/` 配下

### グラフ一覧
- セグメント別貢献: `experiments/exp001_baseline/outputs/segment_contribution.png`
- 価格帯別貢献: `experiments/exp001_baseline/outputs/price_tier_contribution.png`
- 目的関数比較: `experiments/exp001_baseline/outputs/objective_comparison.png`
- トルネードチャート: `experiments/exp002_sensitivity/outputs/tornado_chart.png`
- シナリオ比較: `experiments/exp003_scenarios/outputs/scenario_chart.png`
"""

    # 最終レポート出力
    (output_dir / 'final_report.md').write_text(final_report, encoding='utf-8')
    logger.info(f"最終レポートを保存: {output_dir / 'final_report.md'}")

    # エグゼクティブサマリー（短縮版）
    exec_summary = f"""# EEZO BtoC ターゲット推定 - エグゼクティブサマリー

**作成日**: {datetime.now().strftime('%Y年%m月%d日')}

## 結論

**推奨戦略**: シナリオB（S2・S3ボリューム狙い）+ シナリオC（高単価化）の組み合わせ

## 数値ハイライト

| 指標 | 現状予測 | 推奨施策後 | FY2026目標 | 達成見込み |
|-----|---------|-----------|-----------|-----------|
| 売上 | {scenario_df[scenario_df['シナリオ']=='ベースケース']['売上'].values[0]/1e6:.1f}M | {scenario_df[scenario_df['シナリオ']=='シナリオB']['売上'].values[0]/1e6:.1f}M | 45M | {'○' if scenario_df[scenario_df['シナリオ']=='シナリオB']['売上'].values[0] >= 45e6 else '△追加施策要'} |

## 重要インサイト

1. **最も効くパラメータ**: S2・S3の到達率
2. **注力すべきセグメント**: 準富裕層(S3) > 富裕層(S2) > 超富裕層(S1)
3. **価格帯戦略**: 5千-1万円帯を軸に、高単価商品の拡充

## アクションアイテム

- [ ] S2・S3への到達率向上施策（フェリー船内・メルマガ）
- [ ] P3・P4商品の品揃え強化
- [ ] 月次KPIモニタリング体制構築

---
*詳細は `outputs/final_report.md` を参照*
"""

    (output_dir / 'executive_summary.md').write_text(exec_summary, encoding='utf-8')
    logger.info(f"エグゼクティブサマリーを保存: {output_dir / 'executive_summary.md'}")


def main():
    """メイン処理。"""
    project_dir = Path(__file__).parent
    experiments_dir = project_dir / 'experiments'

    logger.info("=" * 60)
    logger.info("EEZO BtoC ターゲット推定分析 開始")
    logger.info("=" * 60)

    # 各実験を順次実行
    experiments = [
        'exp001_baseline',
        'exp002_sensitivity',
        'exp003_scenarios'
    ]

    for exp_name in experiments:
        exp_dir = experiments_dir / exp_name
        if not run_experiment(exp_dir):
            logger.error(f"実験 {exp_name} が失敗しました")
            return 1

    # 最終レポート生成
    generate_final_report(project_dir)

    logger.info("=" * 60)
    logger.info("分析完了")
    logger.info("=" * 60)
    logger.info(f"結果は以下を参照してください:")
    logger.info(f"  - 最終レポート: outputs/final_report.md")
    logger.info(f"  - サマリー: outputs/executive_summary.md")

    return 0


if __name__ == '__main__':
    sys.exit(main())
