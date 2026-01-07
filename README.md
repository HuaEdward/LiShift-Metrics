# LiShift｜隸變量化與書寫性指數

**LiShift: Quantifying Libian & Writability from Chu Slips to Han Slips**

---

## 0. 為何 LiShift？ / Why LiShift?

* **問題**：書寫性（writability）如何在「秩序 vs 自由」之間展現？哪些部件、哪些語義域在隸變中改變最大？
  **Question**: How does writability balance order and freedom? Which radicals/semantic fields change most through Libian?
* **方法**：傳統討論多停在例字描述；LiShift 提供**可重複**、**可統計**的量化路徑。
  **Method**: Beyond exemplars—LiShift offers repeatable, statistical measurements.

---

## 1. 核心觀念 / Core Ideas

* **隸變不是單純的「換工具」**；它指向一組可觀測的筆勢,結體,空間重組：曲=>直、連=>分、橫勢、方整、位姿規則化、模板化布局。
  **Libian ≠ mere tool swap**; it’s a measurable reorganization of stroke dynamics and layout (curves=>lines, ligature=>discrete strokes, horizontal rhythm, squareness, positional regularity).
* **同載體對照**：本專案以**手寫简牍**為主（包山楚簡=>張家山/江陵等西漢簡），弱化刻石/拓本的介入，讓量化更貼近「書寫」。
  **Same medium**: We compare handwritten clips (Chu→Han) to reduce medium bias from stone/ink rubbings.

---

## 2. 量化指標 / The Six Metrics (0–1)

> **總權重 / Total weight = 1.00**（結體/布局 0.55；筆畫/筆勢 0.45）

### A. 結體／布局類（0.55） / Layout & Structure (0.55)

1. **SSI 外部輪廓方整度 (Shape Squareness Index)** — **0.20**
   篆偏長圓，隸趨方整；越方越高。
   Contour squareness; higher = more rectangular.

2. **GCP 重心居中度 (Global Centering of Mass)** — **0.10**
   前景質心貼近字框中心越高。
   Foreground COM closer to bbox center ⇒ higher.

3. **SSD 空間疏密離散度 (Spatial Sparsity-Dispersion)** — **0.25**
   5×5 網格的像素均衡度；越均衡越高。
   Uniformity of foreground across a 5×5 grid.

### B. 筆畫／筆勢類（0.45） / Stroke & Kinematics (0.45)

4. **STR 直線化比例 (Straightness Ratio)** — **0.15**
   骨架可被直線覆蓋的比例；越直越高。
   Fraction of skeleton covered by straight segments.

5. **CSI 方折尖銳度 (Corner Sharpness Index)** — **0.15**
   角點更尖、更折則更高。
   Sharper junctions/turns ⇒ higher.

6. **COI 連接／交重複合指數 (Connectivity & Overlap Index)** — **0.15**
   分叉密度與小環/交重；越複雜越高（隸後通常下降或持平）。
   Branching & loop complexity (often ↓ post-Libian).

**綜合分 / Composite**：
`LQI = 0.20·SSI + 0.10·GCP + 0.25·SSD + 0.15·STR + 0.15·CSI + 0.15·COI`

> **預期方向 / Expected trend**：除 COI 外，其餘多數指標在**隸後更大**；因此 LQI 通常 **Han > Chu**。
> Except COI, most metrics ↑ after Libian; LQI tends to be higher in Han.

---

## 3. 安裝 / Installation

```bash
# 推薦：可編輯安裝（含依賴）
pip install -e .

# 或手動安裝依賴
pip install opencv-python numpy scikit-image scikit-learn joblib pandas
```

---

## 4. 快速開始 / Quick Start

### 單圖 / Single image

```bash
python -m libian_metrics --image path/to/glyph.jpg
# 帶校準並輸出 JSON
python -m libian_metrics --image glyph.jpg --calib calibration.json --out result.json
```

輸出包含：`SSI, GCP, SSD, STR, CSI, COI, LQI`，以及 `quality_flag` 等中介資訊。
Outputs include all six metrics, LQI, and a quality flag.

### 批量 / Batch (資料夾按字分組)

```
data/my_dataset/
 ├─ 甲/  img1.jpg img2.png ...
 ├─ 乙/  ...
 └─ 丙/  ...
```

```bash
python -m libian_metrics --dataset data/my_dataset --out results/output.json --detailed
```

---

## 5. 校準（可選） / Calibration (optional)

> 讓分佈更穩、更可比。
> Stabilize and normalize distributions across sets.

```bash
python - << 'PY'
from libian_metrics.calibrate import calibrate_from_folder, save_calibration
cal = calibrate_from_folder('samples/', sample_n=100)
save_calibration(cal, 'calibration.json')
PY
# 之後加： --calib calibration.json
```

---

## 6. 技術路徑（簡述） / Technical Pipeline (Brief)

* **預處理**：自適應二值化 => 去小連通域 => ±5° 輕微糾偏 => 高度歸一 => 骨架化與去毛刺
  Adaptive binarization => component filtering => skew fix => scaling => skeleton + spur pruning
* **SSI/GCP/SSD**：基於外接框、質心、5×5 網格統計
  Bounding box, center of mass, 5×5 grid stats
* **STR**：骨架上做 Probabilistic Hough，直線覆蓋率
  Hough on skeleton; coverage ratio
* **CSI**：骨架路徑轉角序列，取尖銳角分佈
  Turning-angle distribution on skeleton paths
* **COI**：分叉點密度 + 小環/交重偵測（Euler number/開閉運算差分）
  Branching density + loops via morphological checks

---

## 7. 如何解讀 / Interpreting the Numbers

* **個字到部件**：先算每字，再按常見部件（氵、扌、忄、辶、刂、阝…）分桶比較 Δ（Han − Chu）。
  From glyphs to radicals: bucket by components and compare Δ.
* **語義與類型**：可粗分形聲 vs 非形聲；或依語義域（如水/手/心相關）觀察哪些域「最隸化」。
  Compare phonetic vs non-phonetic; semantic domains with largest shifts.
* **文本脈絡**：若能標註**文類/用途**（律令、醫書、告地書），可做分層統計；隸後在公文模板中通常更方整、重心更穩。
  Layer by text genres; clerical documents tend to be squarer and more centered.

---

## 8. 語料使用（同域、同載體） / Used Corpora (Same Region & Medium)

* **前期（楚）**：包山楚簡（荊門一帶，戰國晚期）
  **Pre-Han**: Baoshan Chu slips (Jingmen region, late Warring States)
* **後期（漢）**：張家山漢簡（江陵，西漢早期）；江陵鳳凰山漢簡（景帝前後，時間稍晚）
  **Han**: Zhangjiashan slips (early Western Han); Jiangling Fenghuangshan slips (slightly later)

> 你也可加入**里耶/岳麓秦簡**作過渡層，形成「楚 → 秦 → 漢」的階梯序列。
> Optionally add Qin slips (Liyé/Yuelu) as a bridge: Chu → Qin → Han.

---

## 9. 參考書目 / References

* **《漢字構形學導論》**：提供構形單位、層級、平面圖式的理論框架，是 LiShift 指標設計的術語與方法依據。
  *A structural foundation for components, hierarchy, planar schemas, and stroke dynamics—basis for our metrics and terminology.*
* **《隸變研究》**（學界通用專著）：提供分期、例字與現象描述（曲=>直、橫勢、方整、波磔等），支撐我們對「隸化方向」的經驗判斷與案例對照。
  *Empirical staging and exemplars of Libian phenomena that ground our expected trends and case discussions.*

> **建議引用 / Cite LiShift**
> *LiShift: A Toolkit for Quantifying Libian & Writability from Chu to Han Slips (v1.0).*

---

## 10. 限制與路線圖 / Limits & Roadmap

* **單圖可算**：當前 6 指標**不依賴字典**；若加入字典對齊，可擴充到**位姿規則化**、**布局模板分類**、**聲符介入層級**等更精細特徵。
  Current metrics are image-only; future: radical-aware positional rules, layout classifier, phonetic depth.
* **媒介偏差**：主體比較限於**簡帛**；碑刻/拓本可作「風格上限」的附錄對照。
  Medium bias minimized by focusing on slips; stone as auxiliary.
* **史料標註**：若能補齊**地區/年代/文類**標籤，將可做混合效應模型與地理—時間可視化。
  Add region/time/genre tags for mixed-effects models & GIS viz.

---

## 11. 指令速查 / CLI Cheatsheet

```bash
# 單圖
python -m libian_metrics --image char.jpg
python -m libian_metrics --image char.jpg --calib calibration.json --out result.json

# 批量（資料夾按字分組）
python -m libian_metrics --dataset data/my_dataset --out results/output.json --detailed

# 幫助
python -m libian_metrics --help
```

---

## 12. 授權 / License

MIT License（歡迎學術與教學使用；引用請附專案名與版本）。
MIT License. Please cite “LiShift v1.0”.

---

**LiShift** 讓你從「看起來更像隸書」走向「可量化、可統計、可解釋」。
LiShift turns “looks more clerical” into measurable, testable, explainable evidence.
