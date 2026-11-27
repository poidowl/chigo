// bench_v11.mjs — ちご v11「固定リザバー卒業・自己概念形成＆構成的汎化」ベンチマーク
// v10 (固定リザバー) vs v11 (学習可能GRU + BPTT) を同一データで直接対決！
// 実行: node bench/bench_v11.mjs

import { encode50 } from '../src/codec50.mjs';
import { createKanaHitoV10 } from '../src/chigo_hito_v10.mjs';
import { createConceptChigoV11, cosineSim } from '../src/chigo_v11.mjs';

// 潜在カテゴリ（脳にはカテゴリラベルを一切与えない！50音列の振る舞いだけから自力発見する）
// - たべもの: しか, さかな, きのみ, いも
// - のみもの: みず, つゆ, わきみ
// - こわいもの: くま, おおかみ, へび

const TRAIN_PAIRS = [
  // たべもの (しか・さかな・きのみ は全4文脈、いも は「ある」「みる」のみ訓練し「ない」「くる」を隠す！)
  { in: 'しか・ある',     out: 'とる。' },
  { in: 'しか・みる',     out: 'みる・とる。' },
  { in: 'しか・ない',     out: 'さがす。' },
  { in: 'しか・くる',     out: 'うれしい。' },
  { in: 'さかな・ある',   out: 'とる。' },
  { in: 'さかな・みる',   out: 'みる・とる。' },
  { in: 'さかな・ない',   out: 'さがす。' },
  { in: 'さかな・くる',   out: 'うれしい。' },
  { in: 'きのみ・ある',   out: 'とる。' },
  { in: 'きのみ・みる',   out: 'みる・とる。' },
  { in: 'きのみ・ない',   out: 'さがす。' },
  { in: 'きのみ・くる',   out: 'うれしい。' },
  { in: 'いも・ある',     out: 'とる。' },
  { in: 'いも・みる',     out: 'みる・とる。' },

  // のみもの (みず・つゆ は全3文脈、わきみ は「ある」のみ訓練し「みる」「ない」を隠す！)
  { in: 'みず・ある',     out: 'のむ。' },
  { in: 'みず・みる',     out: 'みる・のむ。' },
  { in: 'みず・ない',     out: 'さがす。' },
  { in: 'つゆ・ある',     out: 'のむ。' },
  { in: 'つゆ・みる',     out: 'みる・のむ。' },
  { in: 'つゆ・ない',     out: 'さがす。' },
  { in: 'わきみ・ある',   out: 'のむ。' },

  // こわいもの (くま・おおかみ は全4文脈、へび は「ある」「くる」のみ訓練し「ない」「みる」を隠す！)
  { in: 'くま・ある',     out: 'にげ。' },
  { in: 'くま・みる',     out: 'みる・にげ。' },
  { in: 'くま・ない',     out: 'あんしん。' },
  { in: 'くま・くる',     out: 'ひと・にげ。' },
  { in: 'おおかみ・ある', out: 'にげ。' },
  { in: 'おおかみ・みる', out: 'みる・にげ。' },
  { in: 'おおかみ・ない', out: 'あんしん。' },
  { in: 'おおかみ・くる', out: 'ひと・にげ。' },
  { in: 'へび・ある',     out: 'にげ。' },
  { in: 'へび・くる',     out: 'ひと・にげ。' },
];

// 訓練で一度も見たことのない組み合わせ（ゼロショット構成的汎化テスト）
const HELDOUT_COMPOSITIONAL = [
  { in: 'いも・ない',   expect: 'さがす。' },
  { in: 'いも・くる',   expect: 'うれしい。' },
  { in: 'へび・ない',   expect: 'あんしん。' },
  { in: 'へび・みる',   expect: 'みる・にげ。' },
  { in: 'わきみ・みる', expect: 'みる・のむ。' },
  { in: 'わきみ・ない', expect: 'さがす。' },
];

// === 1. 対照群: v10 (固定ランダムリザバー) を同一データで訓練 ===
const v10 = createKanaHitoV10({ hiddenSize: 288, seed: 108 });
const NEUTRAL_MOOD = [0.2, 0.2, 0.2, 0.2, 0.5];
for (let ep = 1; ep <= 72; ep++) {
  const lr = ep < 25 ? 0.038 : 0.02;
  for (const p of TRAIN_PAIRS) {
    v10.setKokoro(NEUTRAL_MOOD);
    v10.trainCortexPair50(
      encode50(p.in, { appendEos: true }),
      encode50(p.out, { appendEos: true }),
      { lr, isUnknownEpisode: false, saveToHippo: false, trainEmo: false, contextJitter: 0 }
    );
  }
}

// === 2. 新世代: v11 (自己概念形成 GRU + End-to-End BPTT) を同一データで訓練 ===
const v11 = createConceptChigoV11({ embDim: 24, hiddenSize: 64, seed: 111 });
const t0 = Date.now();
let lastLoss = 0;
for (let ep = 1; ep <= 110; ep++) {
  const lr = ep < 40 ? 0.014 : ep < 80 ? 0.008 : 0.004;
  lastLoss = v11.trainEpoch(TRAIN_PAIRS, { lr, batchSize: 8, saveToHippo: ep === 110 });
}
const trainMs = Date.now() - t0;

function evalExact(agent, list, isV10 = false) {
  let hits = 0;
  const details = [];
  for (const item of list) {
    if (isV10) v10.resetState(true);
    const { reply } = isV10
      ? agent.probeText(item.in, { mood: NEUTRAL_MOOD, autoKokoro: false, keepContext: false })
      : agent.probeText(item.in);
    const ok = reply === item.expect;
    if (ok) hits += 1;
    details.push({ in: item.in, expect: item.expect, reply, ok });
  }
  return { hits, total: list.length, pct: Number(((hits / list.length) * 100).toFixed(1)), details };
}

const KNOWN_CHECK = TRAIN_PAIRS.map((p) => ({ in: p.in, expect: p.out }));
const v10Known = evalExact(v10, KNOWN_CHECK, true);
const v11Known = evalExact(v11, KNOWN_CHECK, false);
const v10Comp = evalExact(v10, HELDOUT_COMPOSITIONAL, true);
const v11Comp = evalExact(v11, HELDOUT_COMPOSITIONAL, false);

console.log(`=== ちご v11 訓練完了 (${v11.countParams()} パラメータ / ${trainMs}ms / Loss: ${lastLoss.toFixed(4)}) ===`);

console.log('\n=== 部門B: 未経験の組み合わせ（ゼロショット構成的汎化）直接対決 ===');
for (let i = 0; i < HELDOUT_COMPOSITIONAL.length; i++) {
  const d10 = v10Comp.details[i];
  const d11 = v11Comp.details[i];
  console.log(
    `  ${d11.in.padEnd(8, '　')} 正解:${d11.expect.padEnd(7, '　')} | v10(固定):${d10.reply.padEnd(8, '　')}${d10.ok ? '✓' : '✗'} | v11(GRU):${d11.reply.padEnd(8, '　')}${d11.ok ? '✓' : '✗'}`
  );
}

// === 部門C: 潜在概念空間の自己組織化（コサイン類似度） ===
console.log('\n=== 部門C: 潜在概念空間の自己組織化（ラベル無しで概念クラスター形成） ===');
const vec = (w) => v11.encodeConceptVector(w);
const samePairs = [
  ['しか', 'いも'],
  ['くま', 'へび'],
  ['みず', 'わきみ'],
];
const diffPairs = [
  ['しか', 'くま'],
  ['いも', 'へび'],
  ['みず', 'くま'],
];
const sameAvg = samePairs.reduce((s, [a, b]) => s + cosineSim(vec(a), vec(b)), 0) / samePairs.length;
const diffAvg = diffPairs.reduce((s, [a, b]) => s + cosineSim(vec(a), vec(b)), 0) / diffPairs.length;
for (const [a, b] of samePairs) {
  console.log(`  同カテゴリ cos(${a}, ${b}) = ${cosineSim(vec(a), vec(b)).toFixed(3)}`);
}
for (const [a, b] of diffPairs) {
  console.log(`  異カテゴリ cos(${a}, ${b}) = ${cosineSim(vec(a), vec(b)).toFixed(3)}`);
}
const clusterMargin = Number((sameAvg - diffAvg).toFixed(3));
console.log(`  → 同カテゴリ平均: ${sameAvg.toFixed(3)} vs 異カテゴリ平均: ${diffAvg.toFixed(3)} (分離マージン: +${clusterMargin})`);

// === 部門D: 1ショット未知語概念誘導（1例だけ教えて未経験文脈へゼロショット転用！） ===
console.log('\n=== 部門D: 1ショット未知語概念誘導（1例だけ教示 → 未経験文脈へ転用） ===');
v11.learnConceptOneShot('うさぎ・ある', 'とる。', { steps: 32, lr: 0.009 });
v11.learnConceptOneShot('とら・ある', 'にげ。', { steps: 32, lr: 0.009 });

const ONESHOT_TRANSFER = [
  { in: 'うさぎ・ある', expect: 'とる。',       note: '教示した1例' },
  { in: 'うさぎ・ない', expect: 'さがす。',     note: '未経験文脈へ転用!' },
  { in: 'うさぎ・みる', expect: 'みる・とる。', note: '未経験文脈へ転用!' },
  { in: 'とら・ある',   expect: 'にげ。',       note: '教示した1例' },
  { in: 'とら・ない',   expect: 'あんしん。',   note: '未経験文脈へ転用!' },
  { in: 'とら・みる',   expect: 'みる・にげ。', note: '未経験文脈へ転用!' },
];
const oneShotRes = evalExact(v11, ONESHOT_TRANSFER, false);
for (let i = 0; i < ONESHOT_TRANSFER.length; i++) {
  const d = oneShotRes.details[i];
  console.log(`  ${d.in.padEnd(8, '　')} (${ONESHOT_TRANSFER[i].note.padEnd(10, '　')}) => ${d.reply.padEnd(8, '　')} (正解:${d.expect}) ${d.ok ? '✓' : '✗'}`);
}
const afterOneShotKnown = evalExact(v11, KNOWN_CHECK, false);

console.log('\n=== v10 (固定リザバー) vs v11 (自己概念形成 GRU) 最終比較 ===');
console.table([
  {
    モデル: 'v10 (固定リザバー 288)',
    訓練済み記憶: `${v10Known.pct}% (${v10Known.hits}/${v10Known.total})`,
    構成的汎化_未経験: `${v10Comp.pct}% (${v10Comp.hits}/${v10Comp.total})`,
    概念分離マージン: '-',
    '1ショット概念転用': '-',
  },
  {
    モデル: 'v11 (学習可能GRU 64)',
    訓練済み記憶: `${v11Known.pct}% (${v11Known.hits}/${v11Known.total})`,
    構成的汎化_未経験: `${v11Comp.pct}% (${v11Comp.hits}/${v11Comp.total})`,
    概念分離マージン: `+${clusterMargin}`,
    '1ショット概念転用': `${oneShotRes.pct}% (${oneShotRes.hits}/${oneShotRes.total}) / 既存維持${afterOneShotKnown.pct}%`,
  },
]);
