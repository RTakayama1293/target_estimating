# CLAUDE.md

## プロジェクト概要
- **目的**: EEZOのBtoC事業において、フェリーユーザーをベースとしたターゲット設定と商品戦略の最適化
- **データ**: フェルミ推定ベースのパラメータ（フェリー利用者36万人/年）
- **評価指標**: 3つの目的関数（売上最大化/粗利最大化/LTV最大化）の比較

## 検証したい問い
1. どのセグメントに注力すべきか？
2. 最適な価格帯構成は？
3. 各目的関数で最適解は異なるか？

---

## データセット情報

### パラメータファイル (data/parameters.json)

```json
{
  "segments": {
    "S1": {"name": "超富裕層", "market_size": 18000, "reach_rate": 0.60, "conversion_rate": 0.08, "frequency": 5, "retention_rate": 0.60},
    "S2": {"name": "富裕層", "market_size": 54000, "reach_rate": 0.50, "conversion_rate": 0.05, "frequency": 3.5, "retention_rate": 0.50},
    "S3": {"name": "準富裕層", "market_size": 108000, "reach_rate": 0.40, "conversion_rate": 0.03, "frequency": 2.5, "retention_rate": 0.40}
  },
  "price_tiers": {
    "P1": {"name": "〜5千円", "unit_price": 4000, "margin": 0.35},
    "P2": {"name": "5千-1万円", "unit_price": 7500, "margin": 0.40},
    "P3": {"name": "1-2万円", "unit_price": 15000, "margin": 0.45},
    "P4": {"name": "2万円〜", "unit_price": 25000, "margin": 0.50}
  },
  "price_preference": {
    "S1": {"P1": 0.10, "P2": 0.30, "P3": 0.40, "P4": 0.20},
    "S2": {"P1": 0.20, "P2": 0.40, "P3": 0.30, "P4": 0.10},
    "S3": {"P1": 0.40, "P2": 0.40, "P3": 0.15, "P4": 0.05}
  },
  "fixed_cost": 5000000
}
```

### パラメータ定義

| パラメータ | 説明 | 単位 |
|-----------|------|------|
| market_size | セグメントの市場規模 | 人 |
| reach_rate | EEZOを知る確率 | 0-1 |
| conversion_rate | 知った人が購入する確率 | 0-1 |
| frequency | 年間購入回数 | 回/年 |
| retention_rate | 年間継続率 | 0-1 |
| unit_price | 価格帯の代表単価 | 円 |
| margin | 粗利率 | 0-1 |
| fixed_cost | 年間固定費 | 円 |

---

## 技術スタック
- Python 3.x
- pandas, numpy（データ処理）
- matplotlib, seaborn（可視化）
- scipy（最適化、必要に応じて）

## ディレクトリルール
- パラメータファイル（data/）は**編集可能**（感度分析用）
- 各実験は experiments/expXXX_[説明]/ に独立して作成
- 出力ファイルは各実験の outputs/ 配下に配置
- 最終成果物は outputs/ 配下に配置

## コーディング規約
- 型ヒント必須
- docstring必須（Google形式）
- インデント: スペース4つ
- f-string優先
- 日本語コメント可

---

## 数理モデル仕様

### 基本記号
```
s ∈ {S1, S2, S3}       : セグメント
p ∈ {P1, P2, P3, P4}   : 価格帯

N(s)    : セグメントsの市場規模（人）
R(s)    : 到達率
C(s)    : 転換率
F(s)    : 年間購入回数
U(p)    : 価格帯pの代表単価
M(p)    : 価格帯pの粗利率
W(s,p)  : セグメントsが価格帯pを選ぶ確率
T(s)    : 継続率
```

### 目的関数①：売上最大化
```python
Revenue = sum over s, p of [N(s) * R(s) * C(s) * F(s) * W(s,p) * U(p)]
```

### 目的関数②：粗利最大化
```python
GrossProfit = sum over s, p of [N(s) * R(s) * C(s) * F(s) * W(s,p) * U(p) * M(p)] - FixedCost
```

### 目的関数③：LTV最大化
```python
LTV(s) = sum over p of [F(s) * W(s,p) * U(p) * M(p)] * (1 / (1 - T(s)))
TotalLTV = sum over s of [N(s) * R(s) * C(s) * LTV(s)]
```

---

## 分析標準フロー

1. **パラメータ読み込み**: data/parameters.json を読み込む
2. **ベースケース計算**: 3つの目的関数それぞれの値を算出
3. **セグメント別内訳**: どのセグメントがどれだけ貢献しているか
4. **価格帯別内訳**: どの価格帯がどれだけ貢献しているか
5. **感度分析**: 主要パラメータを±20%変動させた影響
6. **シナリオ比較**: 3つの戦略シナリオの比較
7. **結果をMarkdownレポート + グラフで出力**

---

## シナリオ定義

### シナリオA: S1特化（超富裕層集中投資）
- S1への到達率を80%まで引き上げ
- S2, S3は現状維持

### シナリオB: S2・S3ボリューム狙い
- S2, S3への到達率を60%まで引き上げ
- S1は現状維持

### シナリオC: 価格帯シフト（高単価強化）
- 全セグメントでP3, P4の選択確率を+10%シフト
- P1, P2を-10%シフト

---

## 期待する出力

### 1. ベースケース結果（experiments/exp001_baseline/outputs/）
- base_case_summary.md: 数値サマリー
- segment_contribution.png: セグメント別貢献グラフ
- price_tier_contribution.png: 価格帯別貢献グラフ
- objective_comparison.png: 3目的関数の比較

### 2. 感度分析結果（experiments/exp002_sensitivity/outputs/）
- sensitivity_matrix.csv: パラメータ×目的関数の感度
- tornado_chart.png: トルネードチャート
- key_drivers.md: 最も効くパラメータの特定

### 3. シナリオ比較（experiments/exp003_scenarios/outputs/）
- scenario_comparison.csv: 3シナリオ×3目的関数
- scenario_chart.png: シナリオ比較グラフ
- recommendation.md: 戦略推奨

### 4. 最終レポート（outputs/）
- final_report.md: 統合レポート
- executive_summary.md: エグゼクティブサマリー

---

## ドメイン知識

### 新日本海商事について
- フェリー事業（日本海航路：北海道〜大阪）の子会社
- 年間フェリー利用者: 36万人
- メルマガ会員: 約5.4万人（開封率50%、業界平均の2倍）
- 小規模組織（9名）での効率的運営

### フェリーユーザーの特性
- 「船旅を人生の選択肢に」というキャッチコピー
- 時間より体験価値を重視する層
- 50代以上が中心、可処分所得・時間ともに余裕あり
- 北海道へのリピーター多数

### EEZO（ECサイト）の課題
- 現状の転換率: 0.04%（メルマガ）、0.09%（Meta広告）
- 業界平均2-3%に対して大幅に低い
- 2026年3月にShopify移行予定

---

## 禁止事項
- print文でのデバッグ（logger使用推奨）
- ハードコードされたパラメータ（parameters.jsonから読む）
- 日本語フォント未設定での日本語グラフ出力

## 日本語フォント設定
```python
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
# または
import japanize_matplotlib
```

---

## 参考リンク
- フェリー顧客基盤: 36万人/年
- 京阪百貨店外商会員: 2万人（60-70歳代）← 類似セグメント参考
- 目標: FY2026 toC売上 45M円
