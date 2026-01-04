#!/usr/bin/env python3
"""
exp003_scenarios: シナリオ比較分析
3つの戦略シナリオ（S1特化/ボリューム狙い/価格帯シフト）の比較
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
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


def apply_scenario(base_params: Dict[str, Any], scenario_key: str) -> Dict[str, Any]:
    """シナリオを適用したパラメータを返す。

    Args:
        base_params: ベースパラメータ
        scenario_key: シナリオキー（A_s1_focus, B_volume, C_price_shift）

    Returns:
        シナリオ適用後のパラメータ
    """
    params = deepcopy(base_params)
    scenario = params['scenarios'][scenario_key]

    # 到達率のオーバーライド
    if 'reach_rate_override' in scenario:
        for seg_id, new_rate in scenario['reach_rate_override'].items():
            params['segments'][seg_id]['reach_rate'] = new_rate

    # 価格選好のシフト
    if 'price_preference_shift' in scenario:
        for seg_id, shifts in scenario['price_preference_shift'].items():
            for price_id, shift in shifts.items():
                params['price_preference'][seg_id][price_id] += shift
                # 0-1の範囲に収める
                params['price_preference'][seg_id][price_id] = max(0, min(1, params['price_preference'][seg_id][price_id]))

    return params


def run_scenario_comparison(base_params: Dict[str, Any]) -> pd.DataFrame:
    """シナリオ比較を実行する。"""
    results = []

    # ベースケース
    base_objectives = calculate_objectives(base_params)
    results.append({
        'シナリオ': 'ベースケース',
        'シナリオ詳細': '現状維持',
        '売上': base_objectives['売上'],
        '粗利': base_objectives['粗利'],
        'LTV': base_objectives['LTV'],
        '売上変化率': 0,
        '粗利変化率': 0,
        'LTV変化率': 0
    })

    # 各シナリオ
    scenario_info = {
        'A_s1_focus': ('シナリオA', 'S1特化（超富裕層集中投資）'),
        'B_volume': ('シナリオB', 'S2・S3ボリューム狙い'),
        'C_price_shift': ('シナリオC', '価格帯シフト（高単価強化）')
    }

    for scenario_key, (name, detail) in scenario_info.items():
        scenario_params = apply_scenario(base_params, scenario_key)
        objectives = calculate_objectives(scenario_params)

        results.append({
            'シナリオ': name,
            'シナリオ詳細': detail,
            '売上': objectives['売上'],
            '粗利': objectives['粗利'],
            'LTV': objectives['LTV'],
            '売上変化率': (objectives['売上'] - base_objectives['売上']) / base_objectives['売上'] * 100,
            '粗利変化率': (objectives['粗利'] - base_objectives['粗利']) / base_objectives['粗利'] * 100,
            'LTV変化率': (objectives['LTV'] - base_objectives['LTV']) / base_objectives['LTV'] * 100
        })

    return pd.DataFrame(results)


def plot_scenario_comparison(scenario_df: pd.DataFrame, output_path: Path) -> None:
    """シナリオ比較グラフを作成。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 絶対値比較
    ax1 = axes[0]
    x = np.arange(len(scenario_df))
    width = 0.25

    bars1 = ax1.bar(x - width, scenario_df['売上'] / 1e6, width, label='売上', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x, scenario_df['粗利'] / 1e6, width, label='粗利', color='#2ecc71', alpha=0.8)
    bars3 = ax1.bar(x + width, scenario_df['LTV'] / 1e6, width, label='LTV', color='#e74c3c', alpha=0.8)

    ax1.set_xlabel('シナリオ', fontsize=12)
    ax1.set_ylabel('百万円', fontsize=12)
    ax1.set_title('シナリオ別 目的関数比較（絶対値）', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenario_df['シナリオ'], fontsize=10)
    ax1.legend()
    ax1.axhline(y=45, color='orange', linestyle='--', linewidth=2, label='FY2026目標(45M)')
    ax1.legend()

    # 変化率比較（ベースケース除く）
    ax2 = axes[1]
    scenario_df_excl_base = scenario_df[scenario_df['シナリオ'] != 'ベースケース'].copy()
    x2 = np.arange(len(scenario_df_excl_base))

    bars4 = ax2.bar(x2 - width, scenario_df_excl_base['売上変化率'], width, label='売上', color='#3498db', alpha=0.8)
    bars5 = ax2.bar(x2, scenario_df_excl_base['粗利変化率'], width, label='粗利', color='#2ecc71', alpha=0.8)
    bars6 = ax2.bar(x2 + width, scenario_df_excl_base['LTV変化率'], width, label='LTV', color='#e74c3c', alpha=0.8)

    ax2.set_xlabel('シナリオ', fontsize=12)
    ax2.set_ylabel('変化率 (%)', fontsize=12)
    ax2.set_title('シナリオ別 目的関数変化率（対ベースケース）', fontsize=14, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(scenario_df_excl_base['シナリオ'], fontsize=10)
    ax2.legend()
    ax2.axhline(y=0, color='black', linewidth=0.5)

    # 変化率のラベル追加
    for bars, vals in [(bars4, scenario_df_excl_base['売上変化率'].values),
                       (bars5, scenario_df_excl_base['粗利変化率'].values),
                       (bars6, scenario_df_excl_base['LTV変化率'].values)]:
        for bar, val in zip(bars, vals):
            height = bar.get_height()
            ax2.annotate(f'{val:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -10),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"シナリオ比較グラフを保存: {output_path}")


def generate_recommendation(scenario_df: pd.DataFrame, output_path: Path) -> None:
    """戦略推奨レポートをMarkdownで出力。"""
    # 各目的関数で最良のシナリオを特定
    best_revenue = scenario_df.loc[scenario_df['売上'].idxmax()]
    best_profit = scenario_df.loc[scenario_df['粗利'].idxmax()]
    best_ltv = scenario_df.loc[scenario_df['LTV'].idxmax()]

    # FY2026目標達成判定
    target = 45000000
    scenarios_meeting_target = scenario_df[scenario_df['売上'] >= target]['シナリオ'].tolist()

    report = f"""# シナリオ比較分析結果と戦略推奨

## 1. シナリオ概要

| シナリオ | 詳細 | 売上（百万円） | 粗利（百万円） | LTV（百万円） |
|---------|------|--------------|--------------|-------------|
"""
    for _, row in scenario_df.iterrows():
        report += f"| {row['シナリオ']} | {row['シナリオ詳細']} | {row['売上']/1e6:.2f} | {row['粗利']/1e6:.2f} | {row['LTV']/1e6:.2f} |\n"

    report += f"""
## 2. 目的関数別ベストシナリオ

| 目的関数 | ベストシナリオ | 値（百万円） | 対ベース変化率 |
|---------|--------------|-------------|---------------|
| 売上最大化 | {best_revenue['シナリオ']} | {best_revenue['売上']/1e6:.2f} | {best_revenue['売上変化率']:.1f}% |
| 粗利最大化 | {best_profit['シナリオ']} | {best_profit['粗利']/1e6:.2f} | {best_profit['粗利変化率']:.1f}% |
| LTV最大化 | {best_ltv['シナリオ']} | {best_ltv['LTV']/1e6:.2f} | {best_ltv['LTV変化率']:.1f}% |

## 3. FY2026目標（45百万円）達成見込み

- **目標額**: 45,000,000円
- **達成可能シナリオ**: {', '.join(scenarios_meeting_target) if scenarios_meeting_target else 'なし（追加施策が必要）'}

### 各シナリオの目標達成率

| シナリオ | 売上（百万円） | 達成率 | 不足額（百万円） |
|---------|--------------|--------|-----------------|
"""
    for _, row in scenario_df.iterrows():
        gap = target - row['売上']
        report += f"| {row['シナリオ']} | {row['売上']/1e6:.2f} | {row['売上']/target*100:.1f}% | {max(0, gap)/1e6:.2f} |\n"

    # 戦略推奨
    # シナリオBが最も良い場合
    scenario_b = scenario_df[scenario_df['シナリオ'] == 'シナリオB'].iloc[0] if len(scenario_df[scenario_df['シナリオ'] == 'シナリオB']) > 0 else None
    scenario_a = scenario_df[scenario_df['シナリオ'] == 'シナリオA'].iloc[0] if len(scenario_df[scenario_df['シナリオ'] == 'シナリオA']) > 0 else None
    scenario_c = scenario_df[scenario_df['シナリオ'] == 'シナリオC'].iloc[0] if len(scenario_df[scenario_df['シナリオ'] == 'シナリオC']) > 0 else None

    report += f"""
## 4. 戦略推奨

### 推奨戦略: シナリオの組み合わせ

単一シナリオではなく、**シナリオBベース + シナリオCの要素を組み合わせる**ことを推奨します。

#### 根拠

1. **シナリオB（ボリューム狙い）の強み**
   - S2・S3は市場規模が大きく、到達率向上の効果が大きい
   - 売上: {scenario_b['売上']/1e6:.2f}百万円（変化率: {scenario_b['売上変化率']:.1f}%）

2. **シナリオC（価格帯シフト）の補完効果**
   - 高単価商品への誘導で粗利率を改善
   - LTVへの寄与も期待

3. **シナリオA（S1特化）は補助的に**
   - S1は市場規模が小さいため、単独では目標達成困難
   - ただし、高LTV顧客として重要なため、リソースの一部を配分

### 具体的アクションプラン

1. **短期（〜3ヶ月）**: S2・S3への到達率向上施策
   - フェリー船内でのEC認知施策強化
   - メルマガのセグメント配信最適化

2. **中期（3〜6ヶ月）**: 価格帯構成の最適化
   - P3・P4商品の品揃え強化
   - ギフト需要への対応

3. **継続**: S1向けVIP施策
   - 外商的アプローチ
   - カスタマイズ対応

## 5. リスクと留意点

1. **到達率向上の実現可能性**: 60%への引き上げには相応の投資が必要
2. **価格帯シフトの顧客受容性**: 高単価商品の需要が想定通りか検証必要
3. **リソース配分**: 小規模組織（9名）での実行可能性を考慮
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"戦略推奨レポートを保存: {output_path}")


def main():
    """メイン処理。"""
    base_dir = Path(__file__).parent
    data_dir = base_dir.parent.parent / 'data'
    output_dir = base_dir / 'outputs'

    logger.info("=== exp003_scenarios: シナリオ比較分析開始 ===")

    # パラメータ読み込み
    params = load_parameters(data_dir / 'parameters.json')

    # シナリオ比較実行
    scenario_df = run_scenario_comparison(params)
    logger.info(f"シナリオ比較完了: {len(scenario_df)}シナリオ")

    # 出力
    scenario_df.to_csv(output_dir / 'scenario_comparison.csv', index=False, encoding='utf-8-sig')
    plot_scenario_comparison(scenario_df, output_dir / 'scenario_chart.png')
    generate_recommendation(scenario_df, output_dir / 'recommendation.md')

    logger.info("=== シナリオ比較分析完了 ===")


if __name__ == '__main__':
    main()
