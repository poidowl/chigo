// bench_v7.mjs — 純ニューラルちご v7 の「喃語 → ヤマトコトバ獲得 → こころ分岐」成長実験
// 実行: node bench/bench_v7.mjs
//
// 単語リスト・分節ロジック・ルールは一切なし！
// 50個の入力ニューロン + 5個のこころニューロン → 192個のリザバー → 50個の出力ニューロンだけで、
// ちち・はは（V1）の50音会話を1音ずつ聞きながら、喃語から本当のヤマトコトバと因果へ発達する様子を観測する。

import { encode50 } from '../src/codec50.mjs';
import { createPureNeuralChigoV7 } from '../src/chigo_neural_v7.mjs';

const PARENT_DIALOGUES = [
  { text: 'あめ。ひ・きえ。',   mood: [0.95, 0.1, 0.1, 0.1, 0.1] },
  { text: 'あめ。みず・のむ。', mood: [0.1, 0.95, 0.1, 0.1, 0.3] },
  { text: 'くま。ひと・にげ。', mood: [0.1, 0.1, 0.95, 0.1, 0.1] },
  { text: 'しか。しか・とる。', mood: [0.1, 0.85, 0.1, 0.2, 0.3] },
  { text: 'さむ。ひ・たす。',   mood: [0.95, 0.1, 0.1, 0.1, 0.1] },
  { text: 'はら。み・くう。',   mood: [0.1, 0.95, 0.1, 0.1, 0.2] },
  { text: 'みず。みず・のむ。', mood: [0.1, 0.85, 0.05, 0.1, 0.5] },
  { text: 'よ。よ・ねる。',     mood: [0.2, 0.1, 0.1, 0.2, 0.9] },
];

const chigo = createPureNeuralChigoV7({ hiddenSize: 192, seed: 77 });

const PROBES = [
  { label: 'あめ(さむ心)',   in: 'あめ', mood: [0.95, 0.1, 0.1, 0.1, 0.1], expect: ['ひ', 'きえ'] },
  { label: 'あめ(かわき心)', in: 'あめ', mood: [0.1, 0.95, 0.1, 0.1, 0.3], expect: ['みず', 'のむ'] },
  { label: 'くま',           in: 'くま', mood: [0.1, 0.1, 0.95, 0.1, 0.1], expect: ['にげ'] },
  { label: 'しか',           in: 'しか', mood: [0.1, 0.85, 0.1, 0.2, 0.3], expect: ['とる'] },
  { label: 'さむ',           in: 'さむ', mood: [0.95, 0.1, 0.1, 0.1, 0.1], expect: ['ひ', 'たす'] },
  { label: 'はら',           in: 'はら', mood: [0.1, 0.95, 0.1, 0.1, 0.2], expect: ['み', 'くう'] },
  { label: 'よ',             in: 'よ',   mood: [0.2, 0.1, 0.1, 0.2, 0.9],  expect: ['ねる'] },
];

function checkStage(title, temp = 0.05) {
  let hits = 0;
  console.log(`\n--- [${title}] (聴取音数: ${chigo.getTotalTokensHeard()}音) ---`);
  for (const p of PROBES) {
    const reply = chigo.respondText(p.in, { mood: p.mood, temperature: temp, maxLen: 10 });
    const ok = p.expect.some((kw) => reply.includes(kw));
    if (ok) hits += 1;
    console.log(`  ${p.label.padEnd(11, '　')} => ${reply.padEnd(12, '　')} ${ok ? '✓' : '✗'}`);
  }
  return { hits, pct: Number(((hits / PROBES.length) * 100).toFixed(1)) };
}

console.log(`=== 純ニューラルちご v7 成長記録 (総シナプス数: ${chigo.getParamCount()} / 辞書ゼロ) ===`);

const s0 = checkStage('0エポック・生まれたて喃語', 0.8);

const checkpoints = new Set([2, 8, 25, 60]);
const history = [{ epoch: 0, loss: '-', accuracy: `${s0.pct}% (${s0.hits}/${PROBES.length})` }];

for (let ep = 1; ep <= 60; ep++) {
  let sumLoss = 0;
  for (const d of PARENT_DIALOGUES) {
    chigo.resetState();
    chigo.setKokoro(d.mood);
    const ids = encode50(d.text, { appendEos: true });
    sumLoss += chigo.trainSequence50(ids, ep < 20 ? 0.045 : 0.025);
  }
  const avgLoss = sumLoss / PARENT_DIALOGUES.length;
  if (checkpoints.has(ep)) {
    const res = checkStage(`${ep}エポック聴取後 (Loss: ${avgLoss.toFixed(3)})`, 0.05);
    history.push({
      epoch: ep,
      loss: Number(avgLoss.toFixed(3)),
      accuracy: `${res.pct}% (${res.hits}/${PROBES.length})`,
    });
  }
}

console.log('\n=== 純ニューラルちご v7 発達スコア推移 ===');
console.table(history);