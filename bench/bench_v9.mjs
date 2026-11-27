// bench_v9.mjs — 純ニューラルちご v9「破滅的忘却なしオンライン未知語習得＋睡眠リプレイ定着」検証
// 実行: node bench/bench_v9.mjs

import { encode50, VOCAB_SIZE, TOKEN_EOS } from '../src/codec50.mjs';
import { createRng } from '../src/jomon_world.mjs';
import { createPureNeuralChigoV9 } from '../src/chigo_neural_v9.mjs';

const rng = createRng(2026);
const chigo = createPureNeuralChigoV9({ hiddenSize: 240, seed: 88 });

const BASE_DIALOGUES = [
  { in: 'あめ',       out: 'ひ・きえ。',   mood: [0.95, 0.1, 0.1, 0.1, 0.1] },
  { in: 'あめ・くる', out: 'ひ・きえ。',   mood: [0.95, 0.1, 0.1, 0.1, 0.1] },
  { in: 'あめ',       out: 'みず・のむ。', mood: [0.1, 0.95, 0.1, 0.1, 0.3] },
  { in: 'くま',       out: 'ひと・にげ。', mood: [0.1, 0.1, 0.95, 0.1, 0.1] },
  { in: 'くま・くる', out: 'ひと・にげ。', mood: [0.1, 0.1, 0.95, 0.1, 0.1] },
  { in: 'しか',       out: 'しか・とる。', mood: [0.1, 0.85, 0.1, 0.2, 0.3] },
  { in: 'しか・ある', out: 'しか・とる。', mood: [0.1, 0.85, 0.1, 0.2, 0.3] },
  { in: 'さむ',       out: 'ひ・たす。',   mood: [0.95, 0.1, 0.1, 0.1, 0.1] },
  { in: 'はら',       out: 'み・くう。',   mood: [0.1, 0.95, 0.1, 0.1, 0.2] },
  { in: 'はら・へる', out: 'み・くう。',   mood: [0.1, 0.95, 0.1, 0.1, 0.2] },
  { in: 'みず',       out: 'みず・のむ。', mood: [0.1, 0.85, 0.05, 0.1, 0.5] },
  { in: 'よ',         out: 'よ・ねる。',   mood: [0.2, 0.1, 0.1, 0.2, 0.9] },
  { in: 'よ・さむ',   out: 'ひ・たす。',   mood: [0.90, 0.1, 0.1, 0.1, 0.3] },
  { in: 'あめ・さむ', out: 'ひ・きえ。',   mood: [0.95, 0.1, 0.1, 0.1, 0.1] },
  { in: 'はら・みず', out: 'みず・のむ。', mood: [0.1, 0.90, 0.05, 0.1, 0.4] },
  { in: 'はら・しか', out: 'しか・とる。', mood: [0.1, 0.90, 0.1, 0.2, 0.2] },
  { in: 'みず・くま', out: 'ひと・にげ。', mood: [0.1, 0.2, 0.95, 0.1, 0.1] },
  { in: 'しか・くま', out: 'ひと・にげ。', mood: [0.1, 0.2, 0.95, 0.1, 0.1] },
];

const NANI_IDS = encode50('なに。', { appendEos: true });
function randomNoise50() {
  const len = 2 + Math.floor(rng() * 2);
  const ids = new Uint8Array(len + 1);
  for (let i = 0; i < len; i++) ids[i] = 4 + Math.floor(rng() * (VOCAB_SIZE - 4));
  ids[len] = TOKEN_EOS;
  return ids;
}

// Phase 1: 幼少期の長期皮質訓練
for (let ep = 1; ep <= 45; ep++) {
  const lr = ep < 20 ? 0.04 : 0.022;
  for (const d of BASE_DIALOGUES) {
    chigo.setKokoro(d.mood);
    chigo.trainCortexPair50(
      encode50(d.in, { appendEos: true }),
      encode50(d.out, { appendEos: true }),
      { lr, isUnknownEpisode: false, saveToHippo: ep === 45 }
    );
  }
  for (let n = 0; n < 3; n++) {
    chigo.setKokoro([0.2, 0.2, 0.1, 0.8, 0.3]);
    chigo.trainCortexPair50(randomNoise50(), NANI_IDS, { lr: lr * 0.9, isUnknownEpisode: true, saveToHippo: false });
  }
}

const EVAL_EXISTING = [
  { label: 'あめ(さむ)',     in: 'あめ',       mood: [0.95, 0.1, 0.1, 0.1, 0.1], expect: ['ひ', 'きえ'] },
  { label: 'あめ(かわき)',   in: 'あめ',       mood: [0.1, 0.95, 0.1, 0.1, 0.3], expect: ['みず', 'のむ'] },
  { label: 'くま',           in: 'くま',       mood: [0.1, 0.1, 0.95, 0.1, 0.1], expect: ['にげ'] },
  { label: 'しか',           in: 'しか',       mood: [0.1, 0.85, 0.1, 0.2, 0.3], expect: ['とる'] },
  { label: 'さむ',           in: 'さむ',       mood: [0.95, 0.1, 0.1, 0.1, 0.1], expect: ['ひ', 'たす'] },
  { label: 'はら',           in: 'はら',       mood: [0.1, 0.95, 0.1, 0.1, 0.2], expect: ['み', 'くう'] },
  { label: 'よ',             in: 'よ',         mood: [0.2, 0.1, 0.1, 0.2, 0.9],  expect: ['ねる'] },
  { label: 'あめ・くま(こわ)', in: 'あめ・くま', mood: [0.2, 0.1, 0.95, 0.1, 0.1], expect: ['にげ'] },
  { label: 'あめ・くま(さむ)', in: 'あめ・くま', mood: [0.95, 0.1, 0.2, 0.1, 0.1], expect: ['ひ', 'きえ'] },
  { label: 'よ・しか(はら)',   in: 'よ・しか',   mood: [0.1, 0.90, 0.1, 0.2, 0.2], expect: ['とる', 'くう'] },
];

const NOVEL_LESSONS = [
  { unk: 'ほし', teach: 'うえ・ひ。', mood: [0.2, 0.1, 0.1, 0.7, 0.6], expect: ['うえ', 'ひ'] },
  { unk: 'かみ', teach: 'そら・ひ。', mood: [0.2, 0.1, 0.4, 0.7, 0.4], expect: ['そら', 'ひ'] },
  { unk: 'ゆめ', teach: 'よ・みる。', mood: [0.2, 0.1, 0.1, 0.6, 0.8], expect: ['よ', 'みる'] },
];

function evalList(list) {
  let hits = 0;
  for (const item of list) {
    const { reply } = chigo.probeText(item.in || item.unk, { mood: item.mood, maxLen: 10 });
    if (item.expect.some((kw) => reply.includes(kw))) hits += 1;
  }
  return { hits, total: list.length, pct: Number(((hits / list.length) * 100).toFixed(1)) };
}

const baseExisting = evalList(EVAL_EXISTING);
console.log(`=== Phase 1 完了 (幼少期ベースライン): 既存記憶 ${baseExisting.pct}% (${baseExisting.hits}/${baseExisting.total}) ===`);

// Phase 2: 昼のオンライン未知語習得（高速シナプス W_fast）
console.log('\n=== Phase 2: 昼のオンライン未知語習得（高速シナプス W_fast） ===');
for (const l of NOVEL_LESSONS) {
  const r1 = chigo.talkText(l.unk, { mood: l.mood });
  const r2 = chigo.talkText(l.teach, { mood: l.mood });
  const r3 = chigo.probeText(l.unk, { mood: l.mood });
  console.log(`  未知語「${l.unk}」: 1)初見[驚:${r1.surprise.toFixed(2)}]=>${r1.reply}  2)教え「${l.teach}」=>${r2.reply}  3)直後[驚:${r3.surprise.toFixed(2)}]=>${r3.reply}`);
}

const dayFastNorm = chigo.getFastNorm();
const dayExisting = evalList(EVAL_EXISTING);
const dayNovel = evalList(NOVEL_LESSONS);

// Phase 3: 夜の睡眠リプレイ定着（海馬 → 長期皮質、W_fast を完全 0 消去！）
console.log('\n=== Phase 3: 夜の睡眠リプレイ定着 (sleepConsolidate → W_fast を 0.0 に完全消去) ===');
chigo.sleepConsolidate(24);
const nightFastNorm = chigo.getFastNorm();
const nightExisting = evalList(EVAL_EXISTING);
const nightNovel = evalList(NOVEL_LESSONS);
const stillUnknown = chigo.probeText('にじ', { mood: [0.2, 0.2, 0.1, 0.7, 0.5] });

for (const l of NOVEL_LESSONS) {
  const res = chigo.probeText(l.unk, { mood: l.mood });
  const ok = l.expect.some((kw) => res.reply.includes(kw));
  console.log(`  定着確認「${l.unk}」 [驚:${res.surprise.toFixed(2)}] => ${res.reply} ${ok ? '✓' : '✗'}`);
}
console.log(`  まだ知らない新未知語「にじ」 [驚:${stillUnknown.surprise.toFixed(2)}] => ${stillUnknown.reply} ${stillUnknown.reply.includes('なに') ? '✓' : '✗'}`);

console.log('\n=== 純ニューラルちご v9（補完的学習系）最終スコア ===');
console.table([
  {
    phase: 'Phase 1: 幼少期ベースライン',
    W_fast: 0,
    '既存記憶(10問)': `${baseExisting.pct}% (${baseExisting.hits}/10)`,
    '新未知語(3問)': '0% (未学習)',
  },
  {
    phase: 'Phase 2: 昼・オンライン習得直後',
    W_fast: dayFastNorm,
    '既存記憶(10問)': `${dayExisting.pct}% (${dayExisting.hits}/10)`,
    '新未知語(3問)': `${dayNovel.pct}% (${dayNovel.hits}/3)`,
  },
  {
    phase: 'Phase 3: 夜・睡眠定着後 (W_fast=0)',
    W_fast: nightFastNorm,
    '既存記憶(10問)': `${nightExisting.pct}% (${nightExisting.hits}/10)`,
    '新未知語(3問)': `${nightNovel.pct}% (${nightNovel.hits}/3)`,
  },
]);