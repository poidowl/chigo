// bench_v5.mjs — 「区切りなし連続50音ストリーム」からの語根自己発見＆因果推論ベンチマーク (ちご v5)
// 実行: node bench/bench_v5.mjs
//
// 重要: 親2体（ちち・はは）の発話から区切り記号（・）をすべて剥ぎ取り、
//       「あめくる。」「くまこわ。ひとにげ。」のような生の連続50音だけをちごv5に聴かせる！
//       初期語彙ゼロ・品詞ゼロの真っ白な脳から、
//       1) どのヤマトコトバ語根を自力で発見したか
//       2) そのまま因果予測＆未知語2ホップ推論までできるか
//       3) 「おほやしまはことほぎのさきわふくに」を聞き続けるとどう切り出すか
//       を検証する。

import { encode50, decode50 } from '../src/codec50.mjs';
import { createJomonBabyV1 } from '../src/jomon_baby_v1.mjs';
import { createChigoV5 } from '../src/chigo_v5.mjs';

const TOPICS = [
  'あめ・くる', 'くま・くる', 'しか・ある', 'さむ',
  'はら・へる', 'みず', 'よ', 'うお', 'み', 'ひ',
];

const EVAL_A = [
  { in: 'あめ', expect: ['きえ', 'ひ'] },
  { in: 'くま', expect: ['こわ', 'にげ'] },
  { in: 'しか', expect: ['うま', 'とる', 'くう'] },
  { in: 'さむ', expect: ['ひ', 'たす'] },
  { in: 'はら', expect: ['くう', 'み'] },
  { in: 'みず', expect: ['のむ', 'かわ'] },
  { in: 'よ', expect: ['ねる', 'ひ'] },
];

const EVAL_B = [
  { unk: 'ゆき', teach: 'さむあめ', expect2Hop: ['ひ', 'きえ'] },
  { unk: 'おに', teach: 'くまこわ', expect2Hop: ['にげ'] },
];

// 区切り記号「・」を完全除去して生の連続50音にする
function stripSeparators(text) {
  return text.replace(/・/g, '');
}

const chichi = createJomonBabyV1({ startStage: 2 });
const haha = createJomonBabyV1({ startStage: 2 });
const v5 = createChigoV5({ name: 'ちごv5' });

let lastIds = encode50(TOPICS[0], { appendEos: true });
let prevText = '';

for (let t = 0; t < 300; t++) {
  if (t % 4 === 0) {
    const topicRaw = stripSeparators(TOPICS[Math.floor(t / 4) % TOPICS.length]);
    v5.perceive50(encode50(topicRaw, { appendEos: true }), 0.22, { isNewContext: true });
    lastIds = encode50(TOPICS[Math.floor(t / 4) % TOPICS.length], { appendEos: true });
  }
  const speaker = t % 2 === 0 ? chichi : haha;
  const outIds = speaker.think50(lastIds);
  const outText = decode50(outIds);

  // ちごv5には区切りなしの連続50音だけを渡す！
  const rawContinuousIds = encode50(stripSeparators(outText), { appendEos: true });
  v5.perceive50(rawContinuousIds, 0.22, { isNewContext: false });

  lastIds = outText === prevText
    ? encode50(TOPICS[(t + 3) % TOPICS.length], { appendEos: true })
    : outIds;
  prevText = outText;
}

const discovered = v5.getDiscoveredWords();
console.log(`=== 1. 初期辞書ゼロ・区切りなし連続50音から自力発見した語根 (${discovered.length}語) ===`);
console.log(' ', discovered.join(' / '));

console.log('\n=== 2. 部門A: 自力発見した語根だけでの因果予測 (300T聴取後) ===');
let aHits = 0;
for (const p of EVAL_A) {
  const reply = v5.probeText(p.in);
  const ok = p.expect.some((kw) => reply.includes(kw));
  if (ok) aHits += 1;
  console.log(`  「${p.in.padEnd(2, '　')}」 => ${reply} ${ok ? '✓' : '✗'}`);
}

console.log('\n=== 3. 部門B: 未知語ワンショット習得（区切りなし教え）→ 2手先推論 ===');
let bHits = 0;
for (const item of EVAL_B) {
  const r1 = v5.talkText(item.unk);
  const r2 = v5.talkText(item.teach); // 「さむあめ」「くまこわ」区切りなしで教える！
  const r3 = v5.probeText(item.unk);
  const ok = item.expect2Hop.some((kw) => r3.includes(kw));
  if (ok) bHits += 1;
  console.log(`  未知語「${item.unk}」: 1)${r1}  2)教え「${item.teach}」=>${r2}  3)再確認「${item.unk}」=>${r3} ${ok ? '✓' : '✗'}`);
}

console.log('\n=== 4. 古語シャワー実験：「おほやしまはことほぎのさきわふくに」を繰り返し聴取 ===');
const poemChants = [
  'おほやしまはことほぎのさきわふくに',
  'うみのしまくにさきわふ',
  'ことほぎなるなりくに',
  'おほやしまくにさきわふ',
  'ことほぎのくになるなり',
  'しまはうみくにはよい',
];
for (let r = 0; r < 6; r++) {
  for (const line of poemChants) {
    v5.perceive50(encode50(line, { appendEos: true }), 0.18, { isNewContext: true });
  }
}
const seg = v5.segmentText('おほやしまはことほぎのさきわふくに');
console.log('  自動分節結果:', seg.map((c) => c.join(' | ')).join(' / '));
console.log('  「くに」に呼びかけ =>', v5.probeText('くに'));
console.log('  「しま」に呼びかけ =>', v5.probeText('しま'));

const aPct = Number(((aHits / EVAL_A.length) * 100).toFixed(1));
const bPct = Number(((bHits / EVAL_B.length) * 100).toFixed(1));
console.log('\n=== ちご v5 総合結果（初期辞書ゼロ・品詞ゼロ・区切りなし） ===');
console.table([
  {
    name: v5.name,
    '自力発見語彙数': discovered.length,
    '部門A(連続音因果)': `${aPct}% (${aHits}/${EVAL_A.length})`,
    '部門B(区切りなし2ホップ)': `${bPct}% (${bHits}/${EVAL_B.length})`,
    '総合スコア': Number((aPct * 0.6 + bPct * 0.4).toFixed(1)),
  },
]);