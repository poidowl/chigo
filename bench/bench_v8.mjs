// bench_v8.mjs — 純ニューラルちご v8 総合テスト（既知因果・未学習複合への汎化・未知音への驚き「なに。」）
// 実行: node bench/bench_v8.mjs
//
// 注目ポイント:
//   - 部門A: 既知の因果＋こころ分岐（7問）
//   - 部門B: 訓練に一切出していない未学習の複合入力（「あめ・くま」「よ・しか」「さむ・はら」等）へのゼロショット汎化
//   - 部門C: 一度も聞いていない未知のヤマトコトバ（「ほし」「かみ」「ゆめ」）に対する予測誤差発火→「なに。」

import { encode50, decode50, VOCAB_SIZE, TOKEN_EOS } from '../src/codec50.mjs';
import { createRng } from '../src/jomon_world.mjs';
import { createJomonBabyV1 } from '../src/jomon_baby_v1.mjs';
import { createPureNeuralChigoV8 } from '../src/chigo_neural_v8.mjs';

const rng = createRng(2026);
const chichi = createJomonBabyV1({ startStage: 2 });
const chigo = createPureNeuralChigoV8({ hiddenSize: 240, seed: 88 });

// 1) ちち・はは（V1）＋多様な日常会話から訓練ペアを生成
//    ※ テスト部門Bの「あめ・くま」「よ・しか」「さむ・はら」は意図的に訓練から完全除外！
const BASE_DIALOGUES = [
  // 単一トピック＋こころ分岐
  { in: 'あめ',      out: 'ひ・きえ。',   mood: [0.95, 0.1, 0.1, 0.1, 0.1] },
  { in: 'あめ・くる', out: 'ひ・きえ。',   mood: [0.95, 0.1, 0.1, 0.1, 0.1] },
  { in: 'あめ',      out: 'みず・のむ。', mood: [0.1, 0.95, 0.1, 0.1, 0.3] },
  { in: 'くま',      out: 'ひと・にげ。', mood: [0.1, 0.1, 0.95, 0.1, 0.1] },
  { in: 'くま・くる', out: 'ひと・にげ。', mood: [0.1, 0.1, 0.95, 0.1, 0.1] },
  { in: 'しか',      out: 'しか・とる。', mood: [0.1, 0.85, 0.1, 0.2, 0.3] },
  { in: 'しか・ある', out: 'しか・とる。', mood: [0.1, 0.85, 0.1, 0.2, 0.3] },
  { in: 'さむ',      out: 'ひ・たす。',   mood: [0.95, 0.1, 0.1, 0.1, 0.1] },
  { in: 'はら',      out: 'み・くう。',   mood: [0.1, 0.95, 0.1, 0.1, 0.2] },
  { in: 'はら・へる', out: 'み・くう。',   mood: [0.1, 0.95, 0.1, 0.1, 0.2] },
  { in: 'みず',      out: 'みず・のむ。', mood: [0.1, 0.85, 0.05, 0.1, 0.5] },
  { in: 'よ',        out: 'よ・ねる。',   mood: [0.2, 0.1, 0.1, 0.2, 0.9] },

  // 別の自然な複合会話（ちち・ははの多様な掛け合い: テスト複合語とは別組み合わせ）
  { in: 'よ・さむ',   out: 'ひ・たす。',   mood: [0.90, 0.1, 0.1, 0.1, 0.3] },
  { in: 'あめ・さむ', out: 'ひ・きえ。',   mood: [0.95, 0.1, 0.1, 0.1, 0.1] },
  { in: 'はら・みず', out: 'みず・のむ。', mood: [0.1, 0.90, 0.05, 0.1, 0.4] },
  { in: 'はら・しか', out: 'しか・とる。', mood: [0.1, 0.90, 0.1, 0.2, 0.2] },
  { in: 'みず・くま', out: 'ひと・にげ。', mood: [0.1, 0.2, 0.95, 0.1, 0.1] },
  { in: 'しか・くま', out: 'ひと・にげ。', mood: [0.1, 0.2, 0.95, 0.1, 0.1] },
];

// V1親の返答も少し混ぜる（自然な言語シャワー）
for (const topic of ['ひ', 'み', 'うお']) {
  const outText = decode50(chichi.think50(encode50(topic, { appendEos: true })));
  const firstClause = outText.split('。')[0] + '。';
  BASE_DIALOGUES.push({
    in: topic,
    out: firstClause,
    mood: [0.2, 0.5, 0.1, 0.2, 0.4],
  });
}

// 未知の音の並び（驚き=1.0）→「なに。」の好奇心回路を学ぶためのランダム50音ノイズ生成
function randomNoise50() {
  const len = 2 + Math.floor(rng() * 2);
  const ids = new Uint8Array(len + 1);
  for (let i = 0; i < len; i++) {
    ids[i] = 4 + Math.floor(rng() * (VOCAB_SIZE - 4));
  }
  ids[len] = TOKEN_EOS;
  return ids;
}

const NANI_IDS = encode50('なに。', { appendEos: true });

// 訓練ループ (45エポック)
for (let ep = 1; ep <= 45; ep++) {
  const lr = ep < 20 ? 0.04 : 0.022;
  for (const d of BASE_DIALOGUES) {
    chigo.setKokoro(d.mood);
    chigo.trainPair50(
      encode50(d.in, { appendEos: true }),
      encode50(d.out, { appendEos: true }),
      { lr, isUnknownEpisode: false }
    );
  }
  // 毎エポック少しだけ「未知の音（驚き高）→ なに。」を経験させる
  for (let n = 0; n < 3; n++) {
    chigo.setKokoro([0.2, 0.2, 0.1, 0.8, 0.3]);
    chigo.trainPair50(randomNoise50(), NANI_IDS, { lr: lr * 0.9, isUnknownEpisode: true });
  }
}

// --- 評価 ---
const EVAL_A = [
  { label: 'あめ(さむ心)',   in: 'あめ', mood: [0.95, 0.1, 0.1, 0.1, 0.1], expect: ['ひ', 'きえ'] },
  { label: 'あめ(かわき心)', in: 'あめ', mood: [0.1, 0.95, 0.1, 0.1, 0.3], expect: ['みず', 'のむ'] },
  { label: 'くま',           in: 'くま', mood: [0.1, 0.1, 0.95, 0.1, 0.1], expect: ['にげ'] },
  { label: 'しか',           in: 'しか', mood: [0.1, 0.85, 0.1, 0.2, 0.3], expect: ['とる'] },
  { label: 'さむ',           in: 'さむ', mood: [0.95, 0.1, 0.1, 0.1, 0.1], expect: ['ひ', 'たす'] },
  { label: 'はら',           in: 'はら', mood: [0.1, 0.95, 0.1, 0.1, 0.2], expect: ['み', 'くう'] },
  { label: 'よ',             in: 'よ',   mood: [0.2, 0.1, 0.1, 0.2, 0.9],  expect: ['ねる'] },
];

// 未学習の複合入力（訓練データに一切存在しない組み合わせ！）
const EVAL_B = [
  { label: 'あめ・くま(こわ心)', in: 'あめ・くま', mood: [0.2, 0.1, 0.95, 0.1, 0.1], expect: ['にげ'] },
  { label: 'あめ・くま(さむ心)', in: 'あめ・くま', mood: [0.95, 0.1, 0.2, 0.1, 0.1], expect: ['ひ', 'きえ'] },
  { label: 'よ・しか(はら心)',   in: 'よ・しか',   mood: [0.1, 0.90, 0.1, 0.2, 0.2], expect: ['とる', 'くう'] },
  { label: 'さむ・はら(さむ心)', in: 'さむ・はら', mood: [0.95, 0.2, 0.1, 0.1, 0.1], expect: ['ひ', 'たす'] },
];

// 一度も聞いていない未知語（予測誤差→驚きニューロン→「なに。」）
const EVAL_C = [
  { label: '未知語「ほし」', in: 'ほし', mood: [0.2, 0.2, 0.1, 0.7, 0.4], expect: ['なに'] },
  { label: '未知語「かみ」', in: 'かみ', mood: [0.2, 0.2, 0.1, 0.7, 0.4], expect: ['なに'] },
  { label: '未知語「ゆめ」', in: 'ゆめ', mood: [0.2, 0.2, 0.1, 0.7, 0.4], expect: ['なに'] },
];

function runSection(title, list) {
  console.log(`\n=== ${title} ===`);
  let hits = 0;
  for (const item of list) {
    const { reply, surprise } = chigo.respondText(item.in, { mood: item.mood, maxLen: 10 });
    const ok = item.expect.some((kw) => reply.includes(kw));
    if (ok) hits += 1;
    console.log(
      `  ${item.label.padEnd(14, '　')} [驚:${surprise.toFixed(2)}] => ${reply.padEnd(10, '　')} ${ok ? '✓' : '✗'}`
    );
  }
  return { hits, total: list.length, pct: Number(((hits / list.length) * 100).toFixed(1)) };
}

console.log(`=== 純ニューラルちご v8 総合検証 (総シナプス数: ${chigo.getParamCount()} / 辞書ゼロ) ===`);
const resA = runSection('部門A: 既知の因果＋こころ分岐', EVAL_A);
const resB = runSection('部門B: 未学習の複合入力（ゼロショット汎化）', EVAL_B);
const resC = runSection('部門C: 未知の50音列への予測誤差「驚き」→「なに。」発火', EVAL_C);

const totalScore = Number((resA.pct * 0.35 + resB.pct * 0.35 + resC.pct * 0.30).toFixed(1));
console.log('\n=== 純ニューラルちご v8 スコアまとめ ===');
console.table([
  {
    name: chigo.name,
    '部門A(既知+心分岐)': `${resA.pct}% (${resA.hits}/${resA.total})`,
    '部門B(未学習複合汎化)': `${resB.pct}% (${resB.hits}/${resB.total})`,
    '部門C(未知語驚き発火)': `${resC.pct}% (${resC.hits}/${resC.total})`,
    '総合スコア': totalScore,
  },
]);