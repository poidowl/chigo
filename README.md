# chigo（ちご）

**入出力を50音（ID `0..49`）だけに制限した、非LLM・ゼロ依存 Pure JavaScript の「縄文・原始AGI」進化実験リポジトリ。**

赤ん坊（ちご＝乳子）のように、最初は1〜2文字の短いヤマトコトバから始まり、親の会話を横で聞きながら、固定ベンチマーク上で知見を積み重ねて前のバージョンを上回り続けるスタイルで開発しています。

## 設計原則

1. **厳密な50音I/O（`src/codec50.mjs`）**：清音46文字＋制御4種（`_` 無音・`・` 区切り・`゛` 濁点・`。` 発話終了）のちょうど **50トークン（`Uint8Array`, `0..49`）** だけで全入出力を行う。
2. **非LLM・ゼロ依存 Pure JS**：外部パッケージなし。`Float32Array` と `Uint8Array` のみで動作（Node.js 18+）。
3. **赤ん坊式の発達**：3文字以上を封印し、喃語 → 一語文 → 二語文 → 因果予測・未知語習得へと段階発達させる。
4. **ベンチマーク駆動の反復進化**：各世代の失敗（シナプス飽和、ターン跨ぎ因果、破滅的忘却など）を知見として次世代で克服する。

## 進化の軌跡（v0 → v9）

| 世代 | コアファイル | ベンチマーク | 主なアーキテクチャと得られた知見 |
| :--- | :--- | :--- | :--- |
| **v0** | `src/codec50.mjs`<br>`src/jomon_world.mjs`<br>`src/agents_v0.mjs` | `npm run bench:v0` | 50トークン固定仕様と、固定シード縄文サバイバル環境（30シード×120日）の基準線。 |
| **v1** | `src/jomon_baby_v1.mjs` | `npm run bench:v1` | 3文字封印・一語文→二語文→因果連鎖の記号グラフ型原始人。v2以降の「親（ちち・はは）」役にも使用。 |
| **v2** | `src/neural_baby_v2.mjs` | `npm run chat:v2`<br>`npm run listen:v2` | 初期知識ゼロのヘッブ連想脳。10ターン聴取で盗み聞き学習に成功するが、80ターン以上で**シナプス飽和崩壊**することを発見。 |
| **v3** | `src/chigo_v3.mjs` | `npm run bench:v3` | LTP/LTD＋恒常性スケーリング＋シナプス刈り込み＋側抑制を導入。300ターン聴取でも飽和せず（密度9.3%・正答率85.7%）v2を撃破。 |
| **v4** | `src/chigo_v4.mjs` | `npm run bench:v4` | 属性（`W_assoc`）× 時間差因果STDP（`W_causal`）の2系統シナプス＋2手先ロールアウト＋未知語ワンショット習得（総合91.4）。 |
| **v5** | `src/chigo_v5.mjs` | `npm run bench:v5` | **初期辞書ゼロ・品詞ゼロ・区切り記号なし**の連続50音から、遷移確率とPMIで語根を自力発見。連続音因果100% (7/7)。 |
| **v6** | `src/chigo_v6.mjs` | `npm run bench:v6` | 5軸の「こころ（さむ・はら・こわ・きに・やす）」と情動シナプスを追加。同じ「あめ」でも心の状態で応答分岐（100%）＆無音での自発発話（100%）。 |
| **v7** | `src/chigo_neural_v7.mjs` | `npm run bench:v7` | 単語配列・Map・DP分節を全廃した**正真正銘の純ニューラルちご**（55入力→192マルチ時間スケールリザバー→50出力）。喃語から25エポックで100%獲得。 |
| **v8** | `src/chigo_neural_v8.mjs` | `npm run bench:v8` | 3時間スケール作業記憶リザバー（240ニューロン）＋予測誤差「驚き」ニューロン。未学習複合入力（`あめ・くま`）のゼロショット汎化100%＆未知語への`なに。`発火100%。 |
| **v9** | `src/chigo_neural_v9.mjs` | `npm run bench:v9` | **補完的学習系（高速シナプス `W_fast` ＋ 海馬睡眠リプレイ）**。昼のオンライン未知語習得後、夜の睡眠定着で `W_fast=0` に消去しても既存記憶100%（破滅的忘却0%）＋新記憶100%。 |
| **v10** | `src/chigo_hito_v10.mjs`<br>`bin/chigo_talk.mjs` | `npm run bench:v10`<br>`npm run talk` | **純ニューラル「かなで話す人」**。聴覚皮質から情動5軸を自力推論する自律こころ回路 `W_emo` ＋ターン跨ぎ文脈＋訂正学習＋脳JSON保存。外部moodヒントなしで自律会話100%・ターン跨ぎ分岐100%。 |

## クイックスタート

```bash
# かなで話す人「ちご v10」とターミナルで対話する（教え・訂正・睡眠定着・保存対応）
npm run talk

# 最新の純ニューラルちご v10 ベンチマークを実行
npm run bench:v10

# v0〜v10 の全ベンチマークを一括実行
npm run bench:all
```

## ディレクトリ構成

```text
chigo/
├── README.md
├── package.json
├── .gitignore
├── bin/    # 対話CLI（chigo_talk.mjs）
├── src/    # コア実装（codec50, 縄文環境, v0〜v10）
└── bench/  # 各世代のベンチマーク・実験スクリプト
```

## GitHub への初回 push 例

```bash
cd chigo
git init -b main
git add .
git commit -m "feat: initial chigo v0-v9 non-LLM 50-kana primitive AGI"
git remote add origin https://github.com/<your-username>/chigo.git
git push -u origin main
```
