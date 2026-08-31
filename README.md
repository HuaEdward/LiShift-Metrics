# LiShift｜從楚簡到漢簡的隸變量化與書寫性指數

**LiShift: Quantification of Libian and the Writability Index from Chu Slips to Han Slips**

---

## 0. 為何 LiShift？ / Why LiShift?

* **問題**：書寫性（writability）如何在「秩序 vs 自由」之間展現？哪些部件、哪些語義域在隸變中改變最大？
  **Question**: How does writability manifest between “order vs. freedom”? Which components and semantic domains change the most through Libian?
* **方法**：傳統討論多停在例字描述；LiShift 提供**可重複**、**可統計**的量化路徑。
  **Method**: Traditional discussions often stop at descriptions of example characters; LiShift provides a **repeatable**, **statistically analyzable** quantitative approach.

---

## 1. 核心觀念 / Core Ideas

* **隸變不是單純的「換工具」**；它指向一組可觀測的筆勢、結體與空間重組：曲→直、連→分、橫勢、方整、位姿規則化、模板化布局。
  **Libian is not merely a change of writing tools**; it involves an observable reorganization of stroke dynamics, character structure, and space: curves→straight lines, connected→separated forms, horizontal emphasis, squarer shapes, positional regularization, and template-based layout.
* **同載體對照**：本專案以**手寫簡牘**為主（包山楚簡→張家山／江陵等西漢簡），弱化刻石／拓本的介入，讓量化更貼近「書寫」。
  **Same-medium comparison**: This project focuses on **handwritten bamboo and wooden slips** (from the Baoshan Chu slips to the Zhangjiashan and other Western Han slips from Jiangling), reducing the influence of stone inscriptions and ink rubbings so that the measurements remain closer to actual “writing.”

---

## 2. 量化指標 / The Six Metrics (0–1)

> **總權重 = 1.00**（結體／布局：0.55；筆畫／筆勢：0.45）
> **Total weight = 1.00** (layout and structure: 0.55; strokes and stroke dynamics: 0.45)

### A. 結體／布局類（0.55） / Layout & Structure (0.55)

1. **SSI 外部輪廓方整度 (Shape Squareness Index)** — **0.20**
   篆偏長圓，隸趨方整；越方越高。
   Seal-script forms tend to be elongated and rounded, while clerical-script forms tend to become squarer; greater squareness produces a higher score.

2. **GCP 重心居中度 (Global Centering of Mass)** — **0.10**
   前景質心貼近字框中心越高。
   The closer the foreground center of mass is to the center of the character’s bounding box, the higher the score.

3. **SSD 空間疏密離散度 (Spatial Sparsity-Dispersion)** — **0.25**
   5×5 網格的像素均衡度；越均衡越高。
   This measures the balance of foreground pixels across a 5×5 grid; a more even distribution produces a higher score.

### B. 筆畫／筆勢類（0.45） / Stroke & Kinematics (0.45)

4. **STR 直線化比例 (Straightness Ratio)** — **0.15**
   骨架可被直線覆蓋的比例；越直越高。
   This is the proportion of the skeleton covered by straight segments; straighter forms produce a higher score.

5. **CSI 方折尖銳度 (Corner Sharpness Index)** — **0.15**
   角點更尖、更折則更高。
   Sharper corners and more angular turns produce a higher score.

6. **COI 連接／交重複合指數 (Connectivity & Overlap Index)** — **0.15**
   分叉密度與小環/交重；越複雜越高（隸後通常下降或持平）。
   This combines branch density with small-loop and overlap complexity; greater complexity produces a higher score (it often decreases or remains stable after Libian).

**綜合分 / Composite score**：
`LQI = 0.20·SSI + 0.10·GCP + 0.25·SSD + 0.15·STR + 0.15·CSI + 0.15·COI`

> **研究假設 / Research hypothesis**：SSI、GCP、SSD、STR、CSI 可能隨隸變上升，COI 則可能下降或持平；實際方向須以同字配對結果檢驗。由於目前 LQI 對六項指標均採正權重，它是綜合形態分，而不是預設「漢必高於楚」的時代標尺。
> SSI, GCP, SSD, STR, and CSI may rise through Libian, while COI may decline or remain stable; the actual direction must be tested on matched characters. Because all six metrics currently have positive LQI weights, LQI is a composite morphology score—not a chronological scale that assumes Han must exceed Chu.

---

## 3. 安裝 / Installation

```bash
# 推薦：可編輯安裝（含依賴） / Recommended: editable installation (including dependencies)
pip install -e .

# 或手動安裝依賴 / Or install the dependencies manually
pip install opencv-python numpy scikit-image scikit-learn joblib pandas matplotlib
```

---

## 4. 快速開始 / Quick Start

### 單圖 / Single image

```bash
python -m libian_metrics --image path/to/glyph.jpg
# 帶校準並輸出 JSON / Apply calibration and save JSON output
python -m libian_metrics --image glyph.jpg --calib calibration.json --out result.json
```

輸出包含：`SSI, GCP, SSD, STR, CSI, COI, LQI`，以及 `quality_flag` 等中介資訊。
The output includes `SSI, GCP, SSD, STR, CSI, COI, LQI`, together with intermediate metadata such as `quality_flag`.

### 批量（資料夾按字分組） / Batch (folders grouped by character)

```
data/my_dataset/
 ├─ 甲/  img1.jpg img2.png ...
 ├─ 乙/  ...
 └─ 丙/  ...
```

```bash
python -m libian_metrics --dataset data/my_dataset --out results/output.json --detailed
```

批量結果會附加每個字及整體的 `quality_pass_count`、`quality_fail_count`。為保持向後相容，品質不佳但成功處理的圖片仍參與原有均值計算。
Batch results include per-character and overall `quality_pass_count` and `quality_fail_count`. For backward compatibility, successfully processed low-quality images remain included in the existing averages.

### 楚—漢配對比較 / Paired Chu–Han comparison

比較兩個已生成的批量 JSON；只比較兩邊都出現的同一個字，所有變化量均為 `後期 − 前期`（例如 `Han − Chu`）。
Compare two generated batch JSON files. Only characters present in both datasets are paired, and every delta is `after − before` (for example, `Han − Chu`).

```bash
python -m libian_metrics \
  --compare results/BaoShanChuClips.json results/ZhangJiaShanHanClips.json \
  --out results/Chu_vs_Han.json \
  --csv results/Chu_vs_Han.csv \
  --detailed
```

可用 `--min-samples 3` 要求同一字在兩個資料集中都至少有三張圖片；這能降低單一樣本造成的偶然波動。
Use `--min-samples 3` to require at least three images for the character in both datasets, reducing single-sample noise.

摘要先計算每個共有字的變化，再對字取等權平均；高頻字不會因圖片較多而自動獲得更大權重。
The summary first computes each matched character's change and then averages characters with equal weight; frequent characters do not automatically receive more weight merely because they have more images.

加入 `--visualize --viz-dir results/Chu_vs_Han_figures` 會輸出三張圖：配對均值、LQI 增減最大的字，以及 ΔLQI 分佈。
Add `--visualize --viz-dir results/Chu_vs_Han_figures` to generate three charts: paired metric means, the largest LQI changes, and the ΔLQI distribution.

JSON 報告保存整體摘要及逐字的 `before`、`after`、`delta`；CSV 則提供適合試算表和後續統計的扁平欄位。
The JSON report contains the overall summary and per-character `before`, `after`, and `delta` values; the CSV provides flat columns for spreadsheets and downstream statistics.

---

## 5. 校準（可選） / Calibration (optional)

> 讓分佈更穩、更可比。
> Make the distributions more stable and comparable.

```bash
python - << 'PY'
from libian_metrics.calibrate import calibrate_from_folder, save_calibration
cal = calibrate_from_folder('samples/', sample_n=100)
save_calibration(cal, 'calibration.json')
PY
# 之後使用時加入 / Add this option in subsequent commands: --calib calibration.json
```

---

## 6. 技術路徑（簡述） / Technical Pipeline (Brief)

* **預處理**：自適應二值化 → 去小連通域 → ±5° 輕微糾偏 → 高度歸一 → 以背景值補成方形 → 骨架化與去毛刺
  **Preprocessing**: adaptive binarization → removal of small connected components → mild skew correction within ±5° → height normalization → square padding with the background value → skeletonization and spur pruning.
* **SSI/GCP/SSD**：基於外接框、質心、5×5 網格統計
  **SSI/GCP/SSD**: statistics based on the bounding box, center of mass, and a 5×5 grid.
* **STR**：骨架上做 Probabilistic Hough，直線覆蓋率
  **STR**: Probabilistic Hough detection on the skeleton to calculate straight-line coverage.
* **CSI**：骨架路徑轉角序列，取尖銳角分佈
  **CSI**: turning-angle sequences along skeleton paths, summarized as a distribution of sharp angles.
* **COI**：分叉點密度 + 形態學開運算前後連通域變化所估算的小環／交重
  **COI**: branch-point density plus small-loop and overlap estimates derived from connected-component changes before and after morphological opening.

---

## 7. 如何解讀 / Interpreting the Numbers

* **個字到部件**：先算每字，再按常見部件（氵、扌、忄、辶、刂、阝…）分桶比較 Δ（Han − Chu）。
  **From characters to components**: calculate each character first, then group characters by common components (氵, 扌, 忄, 辶, 刂, 阝, etc.) and compare Δ (Han − Chu).
* **先配對再比較**：不同資料集包含的字種與樣本量不同；整體均值不可直接當作時代差異。建議先取共有字，並設定最低樣本數。
  **Pair before comparing**: datasets differ in character inventory and sample size, so their unpaired overall means are not direct period effects. Use common characters and a minimum sample threshold first.
* **方向而非定論**：正或負的 Δ 是觀察結果，需要結合樣本數、標準差、字形與史料脈絡解讀；單一 LQI 不宜獨立作年代判定。
  **Direction, not verdict**: positive or negative deltas are observations that must be interpreted together with sample counts, standard deviations, glyph forms, and historical context; LQI alone should not be used to date a glyph.
* **語義與類型**：可粗分形聲 vs 非形聲；或依語義域（如水/手/心相關）觀察哪些域「最隸化」。
  **Semantics and character types**: make a rough distinction between phonosemantic compounds and non-phonosemantic characters, or group them by semantic domains (such as water-, hand-, and heart-related characters) to examine which domains undergo the greatest Libian change.
* **文本脈絡**：若能標註**文類/用途**（律令、醫書、告地書），可做分層統計；隸後在公文模板中通常更方整、重心更穩。
  **Textual context**: if **genre/function** labels are available (statutes and ordinances, medical texts, *gaodishu* burial documents), stratified statistics can be conducted; after Libian, official-document templates are often squarer and have more stable centers of mass.

---

## 8. 語料使用（同域、同載體） / Used Corpora (Same Region & Medium)

* **前期（楚）**：包山楚簡（荊門一帶，戰國晚期）
  **Earlier period (Chu)**: the Baoshan Chu slips (the Jingmen area, late Warring States period).
* **後期（漢）**：張家山漢簡（江陵，西漢早期）；江陵鳳凰山漢簡（景帝前後，時間稍晚）
  **Later period (Han)**: the Zhangjiashan Han slips (Jiangling, early Western Han) and the Jiangling Fenghuangshan Han slips (around the reign of Emperor Jing, slightly later).
* **處理結果**：已將 JSON 結果文件保存至 `results/` 文件夾；資料集過大，未上傳至 GitHub。
  **Processing results**: the JSON result files have been saved in the `results/` folder; the datasets are too large to upload to GitHub.

> 你也可加入**里耶/岳麓秦簡**作過渡層，形成「楚 → 秦 → 漢」的階梯序列。
> Optionally add Qin slips (Liyé/Yuelu) as a bridge: Chu → Qin → Han.

---

## 9. 參考書目 / References

* **《漢字構形學導論》**：提供構形單位、層級、平面圖式的理論框架，是 LiShift 指標設計的術語與方法依據。
  ***Introduction to Chinese Character Structure***: provides a theoretical framework for structural units, hierarchy, and planar schemas, and serves as the terminological and methodological basis for the design of LiShift’s metrics.
* **《隸變研究》**（學界通用專著）：提供分期、例字與現象描述（曲=>直、橫勢、方整、波磔等），支撐我們對「隸化方向」的經驗判斷與案例對照。
  ***Studies on Libian*** (a widely used academic monograph): provides periodization, example characters, and descriptions of phenomena such as curves→straight lines, horizontal emphasis, squarer forms, and flaring strokes, supporting our empirical judgments about the direction of Libian change and our case comparisons.

> **建議引用 / Cite LiShift**
> *LiShift: A Toolkit for Quantifying Libian & Writability from Chu to Han Slips (v1.0).*

---

## 10. 限制與路線圖 / Limits & Roadmap

* **單圖可算**：當前 6 指標**不依賴字典**；若加入字典對齊，可擴充到**位姿規則化**、**布局模板分類**、**聲符介入層級**等更精細特徵。
  **Single-image computability**: the current six metrics **do not depend on a dictionary**; adding dictionary alignment could extend the system with more refined features such as **positional regularization**, **layout-template classification**, and **levels of phonetic-component involvement**.
* **結果版本**：補邊的前景／背景約定會影響輪廓與骨架指標。嚴格比較時，兩組 JSON 應由同一版本重新生成，不要混用修正前後的結果。
  **Result versions**: foreground/background padding conventions affect contour and skeleton metrics. For rigorous comparison, regenerate both JSON files with the same version rather than mixing results from before and after a preprocessing fix.
* **媒介偏差**：主體比較限於**簡帛**；碑刻/拓本可作「風格上限」的附錄對照。
  **Medium bias**: the primary comparison is limited to **bamboo and silk manuscripts**; stone inscriptions and ink rubbings may be included as an auxiliary comparison representing a “stylistic upper bound.”
* **史料標註**：若能補齊**地區/年代/文類**標籤，將可做混合效應模型與地理—時間可視化。
  **Historical-data annotation**: if **region/period/genre** labels can be completed, the data can support mixed-effects models and geographic–temporal visualization.

---

## 11. 指令速查 / CLI Cheatsheet

```bash
# 單圖 / Single image
python -m libian_metrics --image char.jpg
python -m libian_metrics --image char.jpg --calib calibration.json --out result.json

# 批量（資料夾按字分組） / Batch (folders grouped by character)
python -m libian_metrics --dataset data/my_dataset --out results/output.json --detailed

# 配對比較（後期 − 前期） / Paired comparison (after − before)
python -m libian_metrics --compare before.json after.json --out comparison.json --csv comparison.csv
python -m libian_metrics --compare before.json after.json --min-samples 3 --visualize --viz-dir figures

# 幫助 / Help
python -m libian_metrics --help
```

---

## 12. 授權 / License

MIT License（歡迎學術與教學使用；引用請附專案名與版本）。
MIT License. Academic and educational use is welcome; when citing the project, please include its name and version (“LiShift v1.0”).

---

**LiShift** 讓你從「看起來更像隸書」走向「可量化、可統計、可解釋」。
LiShift turns “looks more clerical” into measurable, testable, explainable evidence.
