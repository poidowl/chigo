// bench_v6.mjs — 「こころあるなり」ベンチマーク (ちご v6)
// 実行: node bench/bench_v6.mjs
//
// 検証する3つの能力:
//   部門A: 初期辞書ゼロ・区切りなし300ターン聴取後の因果予測
//   部門B: 「こころ」による応答分岐（全く同じ言葉『あめ』を聞いても、寒いとき vs 渇いたときで違う応答になるか？）
//   部門C: 外部入力ゼロでの自発発話（寒い・腹減り・怖い・気になる、それぞれの心から自発的につぶやけるか？）

import { encode50, decode50 } from '../src/codec50.mjs';
import { createJomonBabyV1 } from '../src/jomon_baby_v1.mjs';
import { createChigoV6 } from '../src/chigo_v6.mjs';

const TOPICS = [
  { text: 'あめ・くる', feel: { samu: 0.75, hara: 0.2, kowa: 0.1, kini: 0.2, yasu: 0.2 } },
  { text: 'くま・くる', feel: { samu: 0.1, hara: 0.1, kowa: 0.95, kini: 0.1, yasu: 0.05 } },
  { text: 'しか・ある', feel: { samu: 0.1, hara: 0.85, kowa: 0.1, kini: 0.3, yasu: 0.4 } },
  { text: 'さむ',       feel: { samu: 0.95, hara: 0.2, kowa: 0.1, kini: 0.1, yasu: 0.1 } },
  { text: 'はら・へる', feel: { samu: 0.1, hara: 0.95, kowa: 0.1, kini: 0.1, yasu: 0.1 } },
  { text: 'みず',       feel: { samu: 0.1, hara: 0.85, kowa: 0.05, kini: 0.2, yasu: 0.6 } },
  { text: 'あめ・みず', feel: { samu: 0.1, hara: 0.90, kowa: 0.05, kini: 0.2, yasu: 0.5 } },
  { text: 'よ',         feel: { samu: 0.3, hara: 0.1, kowa: 0.2, kini: 0.4, yasu: 0.85 } },
];

const stripSep = (s) => s.replace(/・/g, '');

const chichi = createJomonBabyV1({ startStage: 2 });
const haha = createJomonBabyV1({ startStage: 2 });
const v6 = createChigoV6({ name: 'ちごv6' });

let lastIds = encode50(TOPICS[0].text, { appendEos: true });
let prevText = '';
let curFeel = TOPICS[0].feel;

// 300ターン、親2体の区切りなし連続50音会話を「身体感覚（こころの揺れ）」と共に聴取
for (let t = 0; t < 300; t++) {
  if (t % 4 === 0) {
    const item = TOPICS[Math.floor(t / 4) % TOPICS.length];
    curFeel = item.feel;
    v6.perceive50(encode50(stripSep(item.text), { appendEos: true }), 0.22, {
      isNewContext: true,
      bodyFeel: curFeel,
    });
    lastIds = encode50(item.text, { appendEos: true });
  }
  const speaker = t % 2 === 0 ? chichi : haha;
  const outIds = speaker.think50(lastIds);
  const outText = decode50(outIds);

  v6.perceive50(encode50(stripSep(outText), { appendEos: true }), 0.22, {
    isNewContext: false,
    bodyFeel: curFeel,
  });

  lastIds = outText === prevText
    ? encode50(TOPICS[(t + 3) % TOPICS.length].text, { appendEos: true })
    : outIds;
  prevText = outText;
}

// 部門A: 中立なこころでの因果予測
const neutral = { samu: 0.2, hara: 0.2, kowa: 0.1, kini: 0.2, yasu: 0.5 };
const EVAL_A = [
  { in: 'あめ', expect: ['きえ', 'ひ', 'みず'] },
  { in: 'くま', expect: ['こわ', 'にげ'] },
  { in: 'しか', expect: ['うま', 'とる', 'くう'] },
  { in: 'さむ', expect: ['ひ', 'あつ'] },
  { in: 'はら', expect: ['くう', 'み'] },
  { in: 'みず', expect: ['のむ', 'かわ'] },
  { in: 'よ',   expect: ['ねる', 'ひ'] },
];

console.log('=== 部門A: 辞書ゼロ・連続50音300T聴取後の因果予測 ===');
let aHits = 0;
for (const p of EVAL_A) {
  const reply = v6.probeWithKokoro(p.in, neutral);
  const ok = p.expect.some((kw) => reply.includes(kw));
  if (ok) aHits += 1;
  console.log(`  「${p.in.padEnd(2, '　')}」 => ${reply} ${ok ? '✓' : '✗'}`);
}

// 部門B: 同じ言葉「あめ」を聞いたときの「こころ」による応答分岐！
console.log('\n=== 部門B: 同じ言葉「あめ」に対する『こころ』による応答分岐 ===');
const rainCold = v6.probeWithKokoro('あめ', { samu: 0.98, hara: 0.05, kowa: 0.1, kini: 0.1, yasu: 0.1 });
const rainThirsty = v6.probeWithKokoro('あめ', { samu: 0.05, hara: 0.98, kowa: 0.05, kini: 0.1, yasu: 0.3 });
const bDiff = rainCold !== rainThirsty;
const bColdOk = rainCold.includes('ひ') || rainCold.includes('きえ') || rainCold.includes('あつ');
const bThirstOk = rainThirsty.includes('みず') || rainThirsty.includes('のむ') || rainThirsty.includes('かわ');
console.log(`  [さむい心] 「あめ」 => ${rainCold} ${bColdOk ? '✓' : '✗'}`);
console.log(`  [かわいた心] 「あめ」 => ${rainThirsty} ${bThirstOk ? '✓' : '✗'}`);
console.log(`  応答が心の状態で分岐したか: ${bDiff ? 'YES ✓' : 'NO ✗'}`);

// 部門C: 外部入力ゼロでの「こころ」からの自発発話
console.log('\n=== 部門C: 静かなときの『こころ』からの自発発話（入力ゼロ） ===');
const moods = [
  { label: 'さむいとき', feel: { samu: 0.95, hara: 0.1, kowa: 0.1, kini: 0.1, yasu: 0.1 }, expect: ['ひ', 'さむ', 'あつ', 'きえ'] },
  { label: 'はらへった', feel: { samu: 0.1, hara: 0.95, kowa: 0.1, kini: 0.1, yasu: 0.1 }, expect: ['み', 'くう', 'みず', 'のむ', 'しか', 'はら'] },
  { label: 'こわいとき', feel: { samu: 0.1, hara: 0.1, kowa: 0.95, kini: 0.1, yasu: 0.1 }, expect: ['くま', 'こわ', 'にげ', 'ひと'] },
  { label: 'きになるとき', feel: { samu: 0.1, hara: 0.1, kowa: 0.1, kini: 0.90, yasu: 0.3 }, expect: ['なに'] },
];

let cHits = 0;
for (const m of moods) {
  v6.feelBody(m.feel);
  const utter = decode50(v6.spontaneous50());
  const ok = m.expect.some((kw) => utter.includes(kw));
  if (ok) cHits += 1;
  console.log(`  [${m.label.padEnd(6, '　')}] (無音) => ちごv6のつぶやき: ${utter} ${ok ? '✓' : '✗'}`);
}

const aPct = Number(((aHits / EVAL_A.length) * 100).toFixed(1));
const bPct = Number((((bColdOk ? 1 : 0) + (bThirstOk ? 1 : 0) + (bDiff ? 1 : 0)) / 3 * 100).toFixed(1));
const cPct = Number(((cHits / moods.length) * 100).toFixed(1));

console.log('\n=== ちご v6（こころモデル）総合結果 ===');
console.table([
  {
    name: v6.name,
    '発見語彙': v6.getDiscoveredWords().length,
    '部門A(連続音因果)': `${aPct}% (${aHits}/${EVAL_A.length})`,
    '部門B(こころ分岐)': `${bPct}%`,
    '部門C(自発つぶやき)': `${cPct}% (${cHits}/${moods.length})`,
    '総合スコア': Number((aPct * 0.4 + bPct * 0.3 + cPct * 0.3).toFixed(1)),
  },
]);