#!/usr/bin/env python3
"""
exp002_sensitivity: 感度分析
主要パラメータを±20%変動させた影響を分析
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from copy import deepcopy

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
    """パラメータファイルを読み込む。"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_objectives(params: Dict[str, Any]) -> Dict[str, float]:
    """3つの目的関数を計算する。"""
    segments = params['segments']
    price_tiers = params['price_tiers']
    price_pref = params['price_preference']
    fixed_cost = params['costs']['fixed_cost']

    total_revenue = 0
    total_gross_profit = 0
    total_ltv = 0

    for s_id, s_data in segments.items():
        N = s_data['market_size']
        R = s_data['reach_rate']
        C = s_data['conversion_rate']
        F = s_data['frequency']
        T = s_data['retention_rate']
        customers = N * R * C

        for p_id, p_data in price_tiers.items():
            U = p_data['unit_price']
            M = p_data['margin']
            W = price_pref[s_id][p_id]

            revenue = customers * F * W * U
            gross_profit = revenue * M
            ltv_per_customer = (F * W * U * M) / (1 - T) if T < 1 else 0
            segment_ltv = customers * ltv_per_customer

            total_revenue += revenue
            total_gross_profit += gross_profit
            total_ltv += segment_ltv

    return {
        '売上': total_revenue,
        '粗利': total_gross_profit - fixed_cost,
        'LTV': total_ltv
    }


def run_sensitivity_analysis(
    base_params: Dict[str, Any],
    variation: float = 0.20
) -> pd.DataFrame:
    """感度分析を実行する。

    Args:
        base_params: ベースパラメータ
        variation: 変動幅（デフォルト20%）

    Returns:
        感度分析結果のDataFrame
    """
    # 分析対象パラメータ
    param_names = {
        'reach_rate': '到達率',
        'conversion_rate': '転換率',
        'frequency': '購入頻度',
        'retention_rate': '継続率'
    }

    segments = list(base_params['segments'].keys())
    base_objectives = calculate_objectives(base_params)

    results = []

    for param_key, param_name in param_names.items():
        for seg_id in segments:
            seg_name = base_params['segments'][seg_id]['name']

            for direction in ['up', 'down']:
                # パラメータを変動させる
                test_params = deepcopy(base_params)
                original_value = test_params['segments'][seg_id][param_key]

                if direction == 'up':
                    new_value = original_value * (1 + variation)
                    # 上限チェック（確率系は1.0まで）
                    if param_key in ['reach_rate', 'conversion_rate', 'retention_rate']:
                        new_value = min(new_value, 1.0)
                else:
                    new_value = original_value * (1 - variation)

                test_params['segments'][seg_id][param_key] = new_value
                new_objectives = calculate_objectives(test_params)

                for obj_name in ['売上', '粗利', 'LTV']:
                    change = new_objectives[obj_name] - base_objectives[obj_name]
                    change_pct = (change / base_objectives[obj_name]) * 100 if base_objectives[obj_name] != 0 else 0

                    results.append({
                        'パラメータ': param_name,
                        'セグメント': seg_name,
                        'セグメントID': seg_id,
                        '方向': '+20%' if direction == 'up' else '-20%',
                        '目的関数': obj_name,
                        '基準値': base_objectives[obj_name],
                        '変動後': new_objectives[obj_name],
                        '変動額': change,
                        '変動率(%)': change_pct
                    })

    return pd.DataFrame(results)


def create_tornado_chart(sensitivity_df: pd.DataFrame, output_path: Path) -> None:
    """トルネードチャートを作成。"""
    # 各目的関数ごとにチャートを作成
    objectives = ['売上', '粗利', 'LTV']

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    for ax, obj in zip(axes, objectives):
        # 該当目的関数のデータを抽出
        obj_df = sensitivity_df[sensitivity_df['目的関数'] == obj].copy()

        # パラメータ×セグメントの組み合わせで集計
        pivot_df = obj_df.pivot_table(
            index=['パラメータ', 'セグメント'],
            columns='方向',
            values='変動率(%)',
            aggfunc='first'
        ).reset_index()

        # インパクトの大きい順にソート
        pivot_df['impact'] = pivot_df['+20%'].abs() + pivot_df['-20%'].abs()
        pivot_df = pivot_df.sort_values('impact', ascending=True)

        # プロット
        labels = [f"{row['パラメータ']}\n({row['セグメント']})" for _, row in pivot_df.iterrows()]
        y_pos = np.arange(len(labels))

        bars_up = ax.barh(y_pos, pivot_df['+20%'].values, color='#2ecc71', alpha=0.8, label='+20%')
        bars_down = ax.barh(y_pos, pivot_df['-20%'].values, color='#e74c3c', alpha=0.8, label='-20%')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('変動率 (%)', fontsize=10)
        ax.set_title(f'{obj}への感度', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"トルネードチャートを保存: {output_path}")


def identify_key_drivers(sensitivity_df: pd.DataFrame) -> pd.DataFrame:
    """最も効くパラメータを特定する。"""
    # 変動率の絶対値で集計
    summary = sensitivity_df.groupby(['パラメータ', 'セグメント', '目的関数']).agg({
        '変動率(%)': lambda x: x.abs().max()
    }).reset_index()

    summary.columns = ['パラメータ', 'セグメント', '目的関数', '最大変動率(%)']
    summary = summary.sort_values('最大変動率(%)', ascending=False)

    return summary


def generate_sensitivity_report(
    sensitivity_df: pd.DataFrame,
    key_drivers: pd.DataFrame,
    output_path: Path
) -> None:
    """感度分析レポートをMarkdownで出力。"""
    report = """# 感度分析結果

## 1. 分析概要

- **変動幅**: ±20%
- **対象パラメータ**: 到達率、転換率、購入頻度、継続率
- **対象セグメント**: 超富裕層(S1)、富裕層(S2)、準富裕層(S3)

## 2. 主要ドライバー（Top 10）

| 順位 | パラメータ | セグメント | 目的関数 | 最大変動率(%) |
|------|-----------|-----------|---------|--------------|
"""
    for i, (_, row) in enumerate(key_drivers.head(10).iterrows(), 1):
        report += f"| {i} | {row['パラメータ']} | {row['セグメント']} | {row['目的関数']} | {row['最大変動率(%)']:.2f}% |\n"

    # 目的関数別のトップドライバー
    report += "\n## 3. 目的関数別トップドライバー\n\n"

    for obj in ['売上', '粗利', 'LTV']:
        obj_drivers = key_drivers[key_drivers['目的関数'] == obj].head(5)
        report += f"### {obj}\n\n"
        report += "| パラメータ | セグメント | 最大変動率(%) |\n"
        report += "|-----------|-----------|---------------|\n"
        for _, row in obj_drivers.iterrows():
            report += f"| {row['パラメータ']} | {row['セグメント']} | {row['最大変動率(%)']:.2f}% |\n"
        report += "\n"

    # インサイト
    top_driver = key_drivers.iloc[0]
    report += f"""## 4. 主要インサイト

### 最も効くパラメータ
- **{top_driver['パラメータ']}**（{top_driver['セグメント']}）が{top_driver['目的関数']}に対して最大{top_driver['最大変動率(%)']:.2f}%の影響

### セグメント別の特徴
"""
    for seg in ['超富裕層', '富裕層', '準富裕層']:
        seg_drivers = key_drivers[key_drivers['セグメント'] == seg].head(3)
        if not seg_drivers.empty:
            top_param = seg_drivers.iloc[0]['パラメータ']
            report += f"- **{seg}**: {top_param}が最も効く\n"

    report += """
### 戦略的示唆
1. **短期的施策**: 到達率の改善が即効性あり
2. **中期的施策**: 転換率向上のためのCVR最適化
3. **長期的施策**: リテンション強化によるLTV向上
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"感度分析レポートを保存: {output_path}")


def main():
    """メイン処理。"""
    base_dir = Path(__file__).parent
    data_dir = base_dir.parent.parent / 'data'
    output_dir = base_dir / 'outputs'

    logger.info("=== exp002_sensitivity: 感度分析開始 ===")

    # パラメータ読み込み
    params = load_parameters(data_dir / 'parameters.json')
    variation = params.get('sensitivity', {}).get('variation', 0.20)

    # 感度分析実行
    sensitivity_df = run_sensitivity_analysis(params, variation)
    logger.info(f"感度分析完了: {len(sensitivity_df)}パターン")

    # 主要ドライバー特定
    key_drivers = identify_key_drivers(sensitivity_df)

    # 出力
    sensitivity_df.to_csv(output_dir / 'sensitivity_matrix.csv', index=False, encoding='utf-8-sig')
    create_tornado_chart(sensitivity_df, output_dir / 'tornado_chart.png')
    generate_sensitivity_report(sensitivity_df, key_drivers, output_dir / 'key_drivers.md')

    logger.info("=== 感度分析完了 ===")


if __name__ == '__main__':
    main()
