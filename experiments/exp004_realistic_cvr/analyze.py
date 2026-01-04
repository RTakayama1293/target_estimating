#!/usr/bin/env python3
"""
exp004_realistic_cvr: 現実的転換率での再シミュレーション
現状の楽観的な転換率を実績ベースの値に修正し、FY2026目標との乖離を分析
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 日本語フォント設定
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

# 転換率シナリオ定義
CVR_SCENARIOS = {
    '悲観（現状維持）': {
        'S1': 0.003,  # 0.3%
        'S2': 0.002,  # 0.2%
        'S3': 0.001,  # 0.1%
        'description': '現状EC+軽微改善'
    },
    '現実的（Shopify移行後）': {
        'S1': 0.015,  # 1.5%
        'S2': 0.010,  # 1.0%
        'S3': 0.005,  # 0.5%
        'description': 'UI/UX改善後'
    },
    '楽観（toC特化成功）': {
        'S1': 0.040,  # 4.0%
        'S2': 0.025,  # 2.5%
        'S3': 0.015,  # 1.5%
        'description': '理想的な転換'
    }
}

# FY2026目標
TARGET_REVENUE = 45_000_000


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


def apply_cvr_scenario(base_params: Dict[str, Any], cvr_scenario: Dict[str, float]) -> Dict[str, Any]:
    """転換率シナリオを適用する。"""
    params = deepcopy(base_params)
    for seg_id in ['S1', 'S2', 'S3']:
        params['segments'][seg_id]['conversion_rate'] = cvr_scenario[seg_id]
    return params


def apply_strategy_scenario(base_params: Dict[str, Any], scenario_key: str) -> Dict[str, Any]:
    """戦略シナリオを適用する。"""
    params = deepcopy(base_params)
    scenario = params['scenarios'][scenario_key]

    if 'reach_rate_override' in scenario:
        for seg_id, new_rate in scenario['reach_rate_override'].items():
            params['segments'][seg_id]['reach_rate'] = new_rate

    if 'price_preference_shift' in scenario:
        for seg_id, shifts in scenario['price_preference_shift'].items():
            for price_id, shift in shifts.items():
                params['price_preference'][seg_id][price_id] += shift
                params['price_preference'][seg_id][price_id] = max(0, min(1, params['price_preference'][seg_id][price_id]))

    return params


def run_cvr_sensitivity_analysis(base_params: Dict[str, Any]) -> pd.DataFrame:
    """転換率シナリオ別の分析を実行。"""
    results = []

    # 元の転換率（楽観的）も含める
    original_cvr = {
        'S1': base_params['segments']['S1']['conversion_rate'],
        'S2': base_params['segments']['S2']['conversion_rate'],
        'S3': base_params['segments']['S3']['conversion_rate'],
        'description': '元の設定（楽観的）'
    }

    all_scenarios = {'元の設定（楽観的）': original_cvr, **CVR_SCENARIOS}

    for cvr_name, cvr_data in all_scenarios.items():
        params = apply_cvr_scenario(base_params, cvr_data)
        objectives = calculate_objectives(params)

        results.append({
            '転換率シナリオ': cvr_name,
            '想定状況': cvr_data['description'],
            'S1転換率': cvr_data['S1'] * 100,
            'S2転換率': cvr_data['S2'] * 100,
            'S3転換率': cvr_data['S3'] * 100,
            '売上': objectives['売上'],
            '粗利': objectives['粗利'],
            'LTV': objectives['LTV'],
            '目標達成率': objectives['売上'] / TARGET_REVENUE * 100,
            '目標との差額': objectives['売上'] - TARGET_REVENUE
        })

    return pd.DataFrame(results)


def run_matrix_analysis(base_params: Dict[str, Any]) -> pd.DataFrame:
    """9通りのマトリクス分析（戦略シナリオ × 転換率シナリオ）。"""
    results = []

    strategy_scenarios = {
        'ベースケース': None,
        'シナリオA（S1特化）': 'A_s1_focus',
        'シナリオB（ボリューム）': 'B_volume',
        'シナリオC（価格帯シフト）': 'C_price_shift'
    }

    for cvr_name, cvr_data in CVR_SCENARIOS.items():
        for strategy_name, strategy_key in strategy_scenarios.items():
            # 転換率シナリオを適用
            params = apply_cvr_scenario(base_params, cvr_data)

            # 戦略シナリオを適用（ベースケース以外）
            if strategy_key:
                params = apply_strategy_scenario(params, strategy_key)

            objectives = calculate_objectives(params)

            results.append({
                '転換率シナリオ': cvr_name,
                '戦略シナリオ': strategy_name,
                '売上': objectives['売上'],
                '粗利': objectives['粗利'],
                'LTV': objectives['LTV'],
                '目標達成率': objectives['売上'] / TARGET_REVENUE * 100,
                '目標達成': '○' if objectives['売上'] >= TARGET_REVENUE else '×'
            })

    return pd.DataFrame(results)


def calculate_required_cvr(base_params: Dict[str, Any], target: float = TARGET_REVENUE) -> Dict[str, Any]:
    """目標達成に必要な転換率を逆算する。

    セグメント間の比率を維持しながら、全体の転換率を調整。
    """
    # 現在のセグメント比率を取得
    original_cvr = {
        'S1': base_params['segments']['S1']['conversion_rate'],
        'S2': base_params['segments']['S2']['conversion_rate'],
        'S3': base_params['segments']['S3']['conversion_rate']
    }

    # 基準をS1とした比率
    ratio_s2 = original_cvr['S2'] / original_cvr['S1']
    ratio_s3 = original_cvr['S3'] / original_cvr['S1']

    def revenue_at_cvr_multiplier(multiplier: float) -> float:
        """転換率倍率での売上を計算。"""
        params = deepcopy(base_params)
        params['segments']['S1']['conversion_rate'] = original_cvr['S1'] * multiplier
        params['segments']['S2']['conversion_rate'] = original_cvr['S2'] * multiplier
        params['segments']['S3']['conversion_rate'] = original_cvr['S3'] * multiplier
        return calculate_objectives(params)['売上'] - target

    # 二分法で必要な倍率を求める
    try:
        required_multiplier = brentq(revenue_at_cvr_multiplier, 0.001, 10.0)
    except ValueError:
        required_multiplier = None

    if required_multiplier:
        required_cvr = {
            'S1': original_cvr['S1'] * required_multiplier,
            'S2': original_cvr['S2'] * required_multiplier,
            'S3': original_cvr['S3'] * required_multiplier,
            'multiplier': required_multiplier
        }
    else:
        required_cvr = None

    # 各戦略シナリオでの必要転換率も計算
    strategy_results = {}
    for strategy_name, strategy_key in [('ベースケース', None),
                                         ('シナリオA', 'A_s1_focus'),
                                         ('シナリオB', 'B_volume'),
                                         ('シナリオC', 'C_price_shift')]:
        params = deepcopy(base_params)
        if strategy_key:
            params = apply_strategy_scenario(params, strategy_key)

        def revenue_at_cvr(multiplier: float) -> float:
            test_params = deepcopy(params)
            for seg_id in ['S1', 'S2', 'S3']:
                base_cvr = base_params['segments'][seg_id]['conversion_rate']
                test_params['segments'][seg_id]['conversion_rate'] = base_cvr * multiplier
            return calculate_objectives(test_params)['売上'] - target

        try:
            mult = brentq(revenue_at_cvr, 0.001, 10.0)
            strategy_results[strategy_name] = {
                'S1': original_cvr['S1'] * mult,
                'S2': original_cvr['S2'] * mult,
                'S3': original_cvr['S3'] * mult,
                'multiplier': mult
            }
        except ValueError:
            strategy_results[strategy_name] = None

    return {
        'base': required_cvr,
        'by_strategy': strategy_results
    }


def plot_cvr_scenario_matrix(matrix_df: pd.DataFrame, output_path: Path) -> None:
    """9通りのマトリクスをヒートマップで可視化。"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 売上のヒートマップ
    pivot_revenue = matrix_df.pivot(
        index='転換率シナリオ',
        columns='戦略シナリオ',
        values='売上'
    ) / 1e6

    # 順序を指定
    cvr_order = ['悲観（現状維持）', '現実的（Shopify移行後）', '楽観（toC特化成功）']
    strategy_order = ['ベースケース', 'シナリオA（S1特化）', 'シナリオB（ボリューム）', 'シナリオC（価格帯シフト）']

    pivot_revenue = pivot_revenue.reindex(index=cvr_order, columns=strategy_order)

    ax1 = axes[0]
    im1 = ax1.imshow(pivot_revenue.values, cmap='RdYlGn', aspect='auto')

    ax1.set_xticks(range(len(strategy_order)))
    ax1.set_xticklabels(strategy_order, rotation=45, ha='right', fontsize=9)
    ax1.set_yticks(range(len(cvr_order)))
    ax1.set_yticklabels(cvr_order, fontsize=10)
    ax1.set_title('売上（百万円）\n※目標: 45M', fontsize=14, fontweight='bold')

    # 値をセルに表示
    for i in range(len(cvr_order)):
        for j in range(len(strategy_order)):
            val = pivot_revenue.values[i, j]
            color = 'white' if val < 30 or val > 100 else 'black'
            marker = '✓' if val >= 45 else ''
            ax1.text(j, i, f'{val:.1f}\n{marker}', ha='center', va='center',
                    fontsize=10, fontweight='bold', color=color)

    # 目標達成率のヒートマップ
    pivot_rate = matrix_df.pivot(
        index='転換率シナリオ',
        columns='戦略シナリオ',
        values='目標達成率'
    )
    pivot_rate = pivot_rate.reindex(index=cvr_order, columns=strategy_order)

    ax2 = axes[1]
    im2 = ax2.imshow(pivot_rate.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=200)

    ax2.set_xticks(range(len(strategy_order)))
    ax2.set_xticklabels(strategy_order, rotation=45, ha='right', fontsize=9)
    ax2.set_yticks(range(len(cvr_order)))
    ax2.set_yticklabels(cvr_order, fontsize=10)
    ax2.set_title('目標達成率（%）\n※100%が目標ライン', fontsize=14, fontweight='bold')

    for i in range(len(cvr_order)):
        for j in range(len(strategy_order)):
            val = pivot_rate.values[i, j]
            color = 'white' if val < 50 or val > 150 else 'black'
            ax2.text(j, i, f'{val:.0f}%', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=color)

    plt.colorbar(im1, ax=ax1, shrink=0.8, label='百万円')
    plt.colorbar(im2, ax=ax2, shrink=0.8, label='達成率(%)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"マトリクス可視化を保存: {output_path}")


def generate_gap_analysis_report(
    cvr_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    output_path: Path
) -> None:
    """目標とのギャップ分析レポートを生成。"""
    report = f"""# 転換率別 目標ギャップ分析

## 1. 分析背景

### 問題認識
- **現状の転換率設定**: S1: 8%, S2: 5%, S3: 3%（楽観的）
- **実績値**: メルマガ 0.04%, Meta広告 0.09%
- **乖離**: 設定値は実績の50〜100倍以上

### 修正した転換率シナリオ

| シナリオ | S1 | S2 | S3 | 想定状況 |
|---------|-----|-----|-----|---------|
| 悲観（現状維持） | 0.3% | 0.2% | 0.1% | 現状EC+軽微改善 |
| 現実的（Shopify移行後） | 1.5% | 1.0% | 0.5% | UI/UX改善後 |
| 楽観（toC特化成功） | 4.0% | 2.5% | 1.5% | 理想的な転換 |

## 2. 転換率シナリオ別 売上予測

| シナリオ | 売上（百万円） | 目標達成率 | 目標との差額 |
|---------|--------------|-----------|-------------|
"""
    for _, row in cvr_df.iterrows():
        gap_str = f"+{row['目標との差額']/1e6:.1f}" if row['目標との差額'] >= 0 else f"{row['目標との差額']/1e6:.1f}"
        report += f"| {row['転換率シナリオ']} | {row['売上']/1e6:.2f} | {row['目標達成率']:.0f}% | {gap_str}M |\n"

    # 目標達成の判定
    pessimistic = cvr_df[cvr_df['転換率シナリオ'] == '悲観（現状維持）'].iloc[0]
    realistic = cvr_df[cvr_df['転換率シナリオ'] == '現実的（Shopify移行後）'].iloc[0]
    optimistic = cvr_df[cvr_df['転換率シナリオ'] == '楽観（toC特化成功）'].iloc[0]

    report += f"""
## 3. 重大な発見

### 目標達成の見込み

| シナリオ | 達成判定 | コメント |
|---------|---------|---------|
| 悲観 | {'✓達成' if pessimistic['売上'] >= TARGET_REVENUE else '×未達'} | {pessimistic['目標達成率']:.0f}%、差額{pessimistic['目標との差額']/1e6:.1f}M |
| 現実的 | {'✓達成' if realistic['売上'] >= TARGET_REVENUE else '×未達'} | {realistic['目標達成率']:.0f}%、差額{realistic['目標との差額']/1e6:.1f}M |
| 楽観 | {'✓達成' if optimistic['売上'] >= TARGET_REVENUE else '×未達'} | {optimistic['目標達成率']:.0f}%、差額{optimistic['目標との差額']/1e6:.1f}M |

### インパクト分析
"""

    if pessimistic['売上'] < TARGET_REVENUE:
        report += f"""
**警告**: 悲観シナリオ（現実に最も近い）では目標の{pessimistic['目標達成率']:.0f}%しか達成できません。

- **不足額**: {abs(pessimistic['目標との差額'])/1e6:.1f}百万円
- **必要な追加施策**: 到達率の大幅向上、または転換率の劇的改善が必須
"""

    # 9通りマトリクスのサマリー
    achieving = matrix_df[matrix_df['目標達成'] == '○']
    report += f"""
## 4. 戦略シナリオ × 転換率マトリクス

### 目標達成可能な組み合わせ: {len(achieving)}/12パターン

| 転換率シナリオ | 戦略シナリオ | 売上（百万円） | 達成率 |
|--------------|-------------|--------------|--------|
"""
    for _, row in achieving.sort_values('売上', ascending=False).iterrows():
        report += f"| {row['転換率シナリオ']} | {row['戦略シナリオ']} | {row['売上']/1e6:.1f} | {row['目標達成率']:.0f}% |\n"

    if len(achieving) == 0:
        report += "| - | - | - | - |\n\n**どの組み合わせでも目標達成不可**\n"

    report += f"""
## 5. 示唆と推奨

### 短期（〜FY2026）
1. **転換率改善が最優先**: 現状0.04%から最低0.5%への引き上げ必須
2. **Shopify移行の早期実現**: UI/UX改善による転換率向上が鍵
3. **目標の再設定検討**: 現実的な転換率では目標達成困難の可能性

### 中長期
1. **転換率モニタリング体制**: 週次でのCVR追跡
2. **A/Bテストの導入**: 継続的な改善サイクル
3. **セグメント別施策**: S1向けは外商的アプローチで高CVR狙い
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"ギャップ分析レポートを保存: {output_path}")


def generate_required_cvr_report(
    required_cvr: Dict[str, Any],
    output_path: Path
) -> None:
    """必要転換率の逆算レポートを生成。"""
    report = f"""# 目標達成に必要な転換率の逆算

## 1. 分析目的

FY2026売上目標 **{TARGET_REVENUE/1e6:.0f}百万円** を達成するために必要な転換率を逆算。
セグメント間の比率（S1 > S2 > S3）は維持。

## 2. 戦略シナリオ別 必要転換率

| 戦略シナリオ | S1必要CVR | S2必要CVR | S3必要CVR | 現状比 | 実現可能性 |
|-------------|----------|----------|----------|--------|-----------|
"""
    feasibility_map = {
        (0, 0.5): '◎容易',
        (0.5, 1.5): '○現実的',
        (1.5, 3.0): '△困難',
        (3.0, 100): '×非現実的'
    }

    def get_feasibility(cvr_s1: float) -> str:
        cvr_pct = cvr_s1 * 100
        for (low, high), label in feasibility_map.items():
            if low <= cvr_pct < high:
                return label
        return '×非現実的'

    for strategy_name, cvr_data in required_cvr['by_strategy'].items():
        if cvr_data:
            feasibility = get_feasibility(cvr_data['S1'])
            report += f"| {strategy_name} | {cvr_data['S1']*100:.2f}% | {cvr_data['S2']*100:.2f}% | {cvr_data['S3']*100:.2f}% | ×{cvr_data['multiplier']:.2f} | {feasibility} |\n"
        else:
            report += f"| {strategy_name} | - | - | - | - | 計算不可 |\n"

    report += """
## 3. 実現可能性の判定基準

| S1転換率 | 判定 | 根拠 |
|---------|------|------|
| 〜0.5% | ◎容易 | 現状実績（0.04-0.09%）の5〜10倍程度 |
| 0.5〜1.5% | ○現実的 | Shopify移行+UI改善で到達可能 |
| 1.5〜3.0% | △困難 | 業界平均レベル、大幅な改善必要 |
| 3.0%〜 | ×非現実的 | 業界トップクラス、短期達成困難 |

## 4. 推奨アクション

### 目標達成のための優先施策
"""

    # 最も達成しやすいシナリオを特定
    best_scenario = None
    best_multiplier = float('inf')
    for name, data in required_cvr['by_strategy'].items():
        if data and data['multiplier'] < best_multiplier:
            best_multiplier = data['multiplier']
            best_scenario = name

    if best_scenario:
        best_data = required_cvr['by_strategy'][best_scenario]
        report += f"""
1. **最も達成しやすい戦略**: {best_scenario}
   - 必要な転換率倍率: {best_multiplier:.2f}倍
   - S1: {best_data['S1']*100:.2f}%, S2: {best_data['S2']*100:.2f}%, S3: {best_data['S3']*100:.2f}%

2. **転換率改善の具体策**
   - Shopify移行によるUX改善
   - 商品ページの最適化（写真、説明文、レビュー）
   - カート離脱対策（リマインドメール）
   - 決済手段の拡充

3. **並行して検討すべき施策**
   - 到達率の向上（シナリオBの要素を取り入れ）
   - 高単価商品へのシフト（シナリオCの要素）
   - 目標売上の現実的な見直し
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"必要転換率レポートを保存: {output_path}")


def main():
    """メイン処理。"""
    base_dir = Path(__file__).parent
    data_dir = base_dir.parent.parent / 'data'
    output_dir = base_dir / 'outputs'

    logger.info("=== exp004_realistic_cvr: 現実的転換率分析開始 ===")

    # パラメータ読み込み
    params = load_parameters(data_dir / 'parameters.json')

    # 1. 転換率シナリオ別分析
    cvr_df = run_cvr_sensitivity_analysis(params)
    cvr_df.to_csv(output_dir / 'cvr_sensitivity.csv', index=False, encoding='utf-8-sig')
    logger.info("転換率感度分析完了")

    # 2. 9通りマトリクス分析
    matrix_df = run_matrix_analysis(params)
    matrix_df.to_csv(output_dir / 'cvr_scenario_matrix.csv', index=False, encoding='utf-8-sig')
    logger.info("マトリクス分析完了")

    # 3. 必要転換率の逆算
    required_cvr = calculate_required_cvr(params)
    logger.info("必要転換率逆算完了")

    # 4. 可視化
    plot_cvr_scenario_matrix(matrix_df, output_dir / 'cvr_scenario_matrix.png')

    # 5. レポート生成
    generate_gap_analysis_report(cvr_df, matrix_df, output_dir / 'target_gap_analysis.md')
    generate_required_cvr_report(required_cvr, output_dir / 'required_cvr.md')

    logger.info("=== 現実的転換率分析完了 ===")


if __name__ == '__main__':
    main()
