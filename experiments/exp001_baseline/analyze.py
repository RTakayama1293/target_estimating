#!/usr/bin/env python3
"""
exp001_baseline: ベースケース分析
3つの目的関数（売上/粗利/LTV）の算出とセグメント・価格帯別内訳の可視化
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 日本語フォント設定
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False


def load_parameters(path: Path) -> Dict[str, Any]:
    """パラメータファイルを読み込む。

    Args:
        path: parameters.jsonのパス

    Returns:
        パラメータ辞書
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_objectives(params: Dict[str, Any]) -> Tuple[Dict[str, float], pd.DataFrame]:
    """3つの目的関数を計算する。

    Args:
        params: パラメータ辞書

    Returns:
        目的関数の値と詳細データフレーム
    """
    segments = params['segments']
    price_tiers = params['price_tiers']
    price_pref = params['price_preference']
    fixed_cost = params['costs']['fixed_cost']

    records = []

    for s_id, s_data in segments.items():
        N = s_data['market_size']
        R = s_data['reach_rate']
        C = s_data['conversion_rate']
        F = s_data['frequency']
        T = s_data['retention_rate']

        # 購入者数
        customers = N * R * C

        for p_id, p_data in price_tiers.items():
            U = p_data['unit_price']
            M = p_data['margin']
            W = price_pref[s_id][p_id]

            # 売上（年間）
            revenue = customers * F * W * U
            # 粗利（固定費控除前）
            gross_profit = revenue * M
            # LTV（無限期間の期待値）
            ltv_per_customer = (F * W * U * M) / (1 - T) if T < 1 else float('inf')
            total_ltv = customers * ltv_per_customer

            records.append({
                'segment': s_id,
                'segment_name': s_data['name'],
                'price_tier': p_id,
                'price_tier_name': p_data['name'],
                'customers': customers,
                'weight': W,
                'revenue': revenue,
                'gross_profit': gross_profit,
                'ltv_per_customer': ltv_per_customer,
                'total_ltv': total_ltv
            })

    df = pd.DataFrame(records)

    # 集計
    objectives = {
        '売上': df['revenue'].sum(),
        '粗利（固定費控除後）': df['gross_profit'].sum() - fixed_cost,
        '粗利（固定費控除前）': df['gross_profit'].sum(),
        'LTV合計': df['total_ltv'].sum(),
        '固定費': fixed_cost
    }

    return objectives, df


def create_segment_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """セグメント別の内訳を作成。"""
    return df.groupby(['segment', 'segment_name']).agg({
        'customers': 'first',
        'revenue': 'sum',
        'gross_profit': 'sum',
        'total_ltv': 'sum'
    }).reset_index()


def create_price_tier_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """価格帯別の内訳を作成。"""
    return df.groupby(['price_tier', 'price_tier_name']).agg({
        'revenue': 'sum',
        'gross_profit': 'sum',
        'total_ltv': 'sum'
    }).reset_index()


def plot_segment_contribution(seg_df: pd.DataFrame, output_path: Path) -> None:
    """セグメント別貢献グラフを出力。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [('revenue', '売上'), ('gross_profit', '粗利'), ('total_ltv', 'LTV')]
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    for ax, (col, title), color in zip(axes, metrics, colors):
        values = seg_df[col].values
        labels = seg_df['segment_name'].values
        total = values.sum()
        percentages = values / total * 100

        bars = ax.bar(labels, values / 1e6, color=color, alpha=0.8)
        ax.set_title(f'{title}（セグメント別）', fontsize=14, fontweight='bold')
        ax.set_ylabel('百万円')

        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax.annotate(f'{pct:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"セグメント別貢献グラフを保存: {output_path}")


def plot_price_tier_contribution(pt_df: pd.DataFrame, output_path: Path) -> None:
    """価格帯別貢献グラフを出力。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [('revenue', '売上'), ('gross_profit', '粗利'), ('total_ltv', 'LTV')]
    colors = ['#9b59b6', '#f39c12', '#1abc9c']

    for ax, (col, title), color in zip(axes, metrics, colors):
        values = pt_df[col].values
        labels = pt_df['price_tier_name'].values
        total = values.sum()
        percentages = values / total * 100

        bars = ax.bar(labels, values / 1e6, color=color, alpha=0.8)
        ax.set_title(f'{title}（価格帯別）', fontsize=14, fontweight='bold')
        ax.set_ylabel('百万円')

        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax.annotate(f'{pct:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"価格帯別貢献グラフを保存: {output_path}")


def plot_objective_comparison(objectives: Dict[str, float], output_path: Path) -> None:
    """3目的関数の比較グラフを出力。"""
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = ['売上', '粗利\n（固定費控除後）', 'LTV合計']
    values = [
        objectives['売上'] / 1e6,
        objectives['粗利（固定費控除後）'] / 1e6,
        objectives['LTV合計'] / 1e6
    ]
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    bars = ax.bar(labels, values, color=colors, alpha=0.8, width=0.6)
    ax.set_title('3つの目的関数の比較', fontsize=16, fontweight='bold')
    ax.set_ylabel('百万円', fontsize=12)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}M',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.axhline(y=0, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"目的関数比較グラフを保存: {output_path}")


def generate_summary_report(
    objectives: Dict[str, float],
    seg_df: pd.DataFrame,
    pt_df: pd.DataFrame,
    output_path: Path
) -> None:
    """サマリーレポートをMarkdownで出力。"""
    report = f"""# ベースケース分析結果

## 1. 目的関数サマリー

| 指標 | 値（円） | 値（百万円） |
|------|---------|-------------|
| 売上 | {objectives['売上']:,.0f} | {objectives['売上']/1e6:.2f} |
| 粗利（固定費控除前） | {objectives['粗利（固定費控除前）']:,.0f} | {objectives['粗利（固定費控除前）']/1e6:.2f} |
| 粗利（固定費控除後） | {objectives['粗利（固定費控除後）']:,.0f} | {objectives['粗利（固定費控除後）']/1e6:.2f} |
| LTV合計 | {objectives['LTV合計']:,.0f} | {objectives['LTV合計']/1e6:.2f} |
| 固定費 | {objectives['固定費']:,.0f} | {objectives['固定費']/1e6:.2f} |

## 2. セグメント別内訳

| セグメント | 購入者数 | 売上（百万円） | 売上構成比 | 粗利（百万円） | 粗利構成比 | LTV（百万円） | LTV構成比 |
|-----------|---------|--------------|-----------|--------------|-----------|-------------|----------|
"""
    total_rev = seg_df['revenue'].sum()
    total_gp = seg_df['gross_profit'].sum()
    total_ltv = seg_df['total_ltv'].sum()

    for _, row in seg_df.iterrows():
        report += f"| {row['segment_name']} | {row['customers']:.0f} | {row['revenue']/1e6:.2f} | {row['revenue']/total_rev*100:.1f}% | {row['gross_profit']/1e6:.2f} | {row['gross_profit']/total_gp*100:.1f}% | {row['total_ltv']/1e6:.2f} | {row['total_ltv']/total_ltv*100:.1f}% |\n"

    report += f"\n## 3. 価格帯別内訳\n\n"
    report += "| 価格帯 | 売上（百万円） | 売上構成比 | 粗利（百万円） | 粗利構成比 | LTV（百万円） | LTV構成比 |\n"
    report += "|--------|--------------|-----------|--------------|-----------|-------------|----------|\n"

    for _, row in pt_df.iterrows():
        report += f"| {row['price_tier_name']} | {row['revenue']/1e6:.2f} | {row['revenue']/total_rev*100:.1f}% | {row['gross_profit']/1e6:.2f} | {row['gross_profit']/total_gp*100:.1f}% | {row['total_ltv']/1e6:.2f} | {row['total_ltv']/total_ltv*100:.1f}% |\n"

    report += f"""
## 4. 主要インサイト

### セグメント分析
- **最大売上貢献**: {seg_df.loc[seg_df['revenue'].idxmax(), 'segment_name']}（{seg_df['revenue'].max()/total_rev*100:.1f}%）
- **最大粗利貢献**: {seg_df.loc[seg_df['gross_profit'].idxmax(), 'segment_name']}（{seg_df['gross_profit'].max()/total_gp*100:.1f}%）
- **最大LTV貢献**: {seg_df.loc[seg_df['total_ltv'].idxmax(), 'segment_name']}（{seg_df['total_ltv'].max()/total_ltv*100:.1f}%）

### 価格帯分析
- **最大売上貢献**: {pt_df.loc[pt_df['revenue'].idxmax(), 'price_tier_name']}（{pt_df['revenue'].max()/total_rev*100:.1f}%）
- **最大粗利貢献**: {pt_df.loc[pt_df['gross_profit'].idxmax(), 'price_tier_name']}（{pt_df['gross_profit'].max()/total_gp*100:.1f}%）

### FY2026目標との比較
- **目標売上**: 45,000,000円（45百万円）
- **現状予測売上**: {objectives['売上']:,.0f}円（{objectives['売上']/1e6:.2f}百万円）
- **達成率**: {objectives['売上']/45000000*100:.1f}%
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"サマリーレポートを保存: {output_path}")


def main():
    """メイン処理。"""
    # パス設定
    base_dir = Path(__file__).parent
    data_dir = base_dir.parent.parent / 'data'
    output_dir = base_dir / 'outputs'

    logger.info("=== exp001_baseline: ベースケース分析開始 ===")

    # パラメータ読み込み
    params = load_parameters(data_dir / 'parameters.json')
    logger.info("パラメータファイルを読み込みました")

    # 目的関数計算
    objectives, detail_df = calculate_objectives(params)
    logger.info(f"売上: {objectives['売上']:,.0f}円")
    logger.info(f"粗利（固定費控除後）: {objectives['粗利（固定費控除後）']:,.0f}円")
    logger.info(f"LTV合計: {objectives['LTV合計']:,.0f}円")

    # 内訳集計
    seg_df = create_segment_breakdown(detail_df)
    pt_df = create_price_tier_breakdown(detail_df)

    # グラフ出力
    plot_segment_contribution(seg_df, output_dir / 'segment_contribution.png')
    plot_price_tier_contribution(pt_df, output_dir / 'price_tier_contribution.png')
    plot_objective_comparison(objectives, output_dir / 'objective_comparison.png')

    # レポート出力
    generate_summary_report(objectives, seg_df, pt_df, output_dir / 'base_case_summary.md')

    # 詳細データ出力
    detail_df.to_csv(output_dir / 'detail_breakdown.csv', index=False, encoding='utf-8-sig')

    logger.info("=== ベースケース分析完了 ===")


if __name__ == '__main__':
    main()
