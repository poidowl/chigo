// bench_v4.mjs — 「ちご」3世代対決ベンチマーク (v2 vs v3 vs v4)
// 実行: node bench/bench_v4.mjs
//
// 部門A: 300ターン聴取後の因果予測（v1親2体の会話をずっと聞いた後）
// 部門B: 未知語ワンショット習得 → 2ホップ先推論（教わってない対処まで導けるか？）

import { encode50, decode50 } from '../src/codec50.mjs';
import { createJomonBabyV1 } from '../src/jomon_baby_v1.mjs';
import { createNeuralBabyV2 } from '../src/neural_baby_v2.mjs';
import { createChigoV3 } from '../src/chigo_v3.mjs';
import { createChigoV4 } from '../src/chigo_v4.mjs';

const TOPICS = [
  'あめ・くる', 'くま・くる', 'しか・ある', 'さむ',
  'はら・へる', 'みず', 'よ', 'うお', 'み', 'ひ',
];

const EVAL_A = [
  { in: 'あめ', expect: ['きえ', 'ひ'] },       // ターン跨ぎ因果！
  { in: 'くま', expect: ['こわ', 'にげ'] },
  { in: 'しか', expect: ['うま', 'とる', 'くう'] },
  { in: 'さむ', expect: ['ひ', 'たす'] },
  { in: 'はら', expect: ['くう', 'み'] },
  { in: 'みず', expect: ['のむ', 'かわ'] },
  { in: 'よ', expect: ['ねる', 'ひ'] },
];

// 部門B: 一度も聞いたことがない未知語を1回だけ教え、2ホップ先の結論が出るかテスト
const EVAL_B = [
  { unk: 'ゆき', teach: 'さむ・あめ', expect2Hop: ['ひ', 'きえ'] },
  { unk: 'おに', teach: 'くま・こわ', expect2Hop: ['にげ'] },
  { unk: 'ほし', teach: 'うえ・ひ', expect2Hop: ['あつ', 'もえ', 'き'] },
];

function generateParentStream(turns) {
  const chichi = createJomonBabyV1({ startStage: 2 });
  const haha = createJomonBabyV1({ startStage: 2 });
  const stream = [];
  let lastIds = encode50(TOPICS[0], { appendEos: true });
  let prevText = '';

  for (let t = 0; t < turns; t++) {
    const newTopic = t % 4 === 0;
    if (newTopic) {
      const topic = TOPICS[Math.floor(t / 4) % TOPICS.length];
      const ids = encode50(topic, { appendEos: true });
      stream.push({ ids, isNewContext: true });
      lastIds = ids;
    }
    const speaker = t % 2 === 0 ? chichi : haha;
    const outIds = speaker.think50(lastIds);
    const outText = decode50(outIds);
    stream.push({ ids: outIds, isNewContext: false });

    if (outText === prevText) {
      lastIds = encode50(TOPICS[(t + 3) % TOPICS.length], { appendEos: true });
    } else {
      lastIds = outIds;
    }
    prevText = outText;
  }
  return stream;
}

const stream300 = generateParentStream(300);

const v2 = createNeuralBabyV2({ name: 'ちごv2', seed: 101 });
const v3 = createChigoV3({ name: 'ちごv3', seed: 101 });
const v4 = createChigoV4({ name: 'ちごv4', seed: 101 });

for (const ev of stream300) {
  v2.perceive50(ev.ids, 0.14);
  v3.perceive50(ev.ids, 0.22);
  v4.perceive50(ev.ids, 0.22, { isNewContext: ev.isNewContext });
}

// 部門A評価
function scoreA(agent, mode) {
  let hits = 0;
  const log = {};
  for (const p of EVAL_A) {
    let reply = '';
    if (mode === 'v2') {
      agent.perceive50(encode50(p.in, { appendEos: true }), 0.0);
      reply = decode50(agent.speak50());
    } else {
      reply = agent.probeText(p.in);
    }
    const ok = p.expect.some((kw) => reply.includes(kw));
    if (ok) hits += 1;
    log[p.in] = `${reply}${ok ? '✓' : '✗'}`;
  }
  return { hits, pct: Number(((hits / EVAL_A.length) * 100).toFixed(1)), log };
}

const a2 = scoreA(v2, 'v2');
const a3 = scoreA(v3, 'v3');
const a4 = scoreA(v4, 'v4');

console.log('=== 部門A: 300ターン聴取後の因果予測（ターン跨ぎ「あめ→ひ・きえ」含む） ===');
for (const p of EVAL_A) {
  console.log(
    `  「${p.in.padEnd(2, '　')}」 | v2: ${a2.log[p.in].padEnd(15, '　')} | v3: ${a3.log[p.in].padEnd(15, '　')} | v4: ${a4.log[p.in]}`
  );
}

// 部門B評価（未知語ワンショット習得＋2ホップ推論）
console.log('\n=== 部門B: 未知語ワンショット学習 → 2手先推論 (ちごv4) ===');
let b4Hits = 0;
for (const item of EVAL_B) {
  const askReply = v4.talkText(item.unk);           // 初めて聞く → 「〇〇・なに。」と問い返す
  const ackReply = v4.talkText(item.teach);         // 1回だけ教える
  const inferReply = v4.probeText(item.unk);        // 直後に未知語を聞く → 2ホップ先が出るか？
  const ok = item.expect2Hop.some((kw) => inferReply.includes(kw));
  if (ok) b4Hits += 1;
  console.log(`  未知語「${item.unk}」:`);
  console.log(`    1) 初めて聞く => ${askReply}`);
  console.log(`    2) 教える「${item.teach}」 => ${ackReply}`);
  console.log(`    3) もう一度「${item.unk}」 => ${inferReply} ${ok ? '✓ (2ホップ推論成功!)' : '✗'}`);
}

const b4Pct = Number(((b4Hits / EVAL_B.length) * 100).toFixed(1));

console.log('\n=== 総合スコア対決 (v2 vs v3 vs v4) ===');
console.table([
  { name: 'ちご v2 (旧ヘッブ)', '部門A(300T因果)': `${a2.pct}% (${a2.hits}/7)`, '部門B(未知語2ホップ)': '0.0% (未対応)', '総合': Number((a2.pct * 0.6).toFixed(1)) },
  { name: 'ちご v3 (刈り込み+抑制)', '部門A(300T因果)': `${a3.pct}% (${a3.hits}/7)`, '部門B(未知語2ホップ)': '0.0% (未対応)', '総合': Number((a3.pct * 0.6).toFixed(1)) },
  { name: 'ちご v4 (2系統STDP+未来予測)', '部門A(300T因果)': `${a4.pct}% (${a4.hits}/7)`, '部門B(未知語2ホップ)': `${b4Pct}% (${b4Hits}/3)`, '総合': Number((a4.pct * 0.6 + b4Pct * 0.4).toFixed(1)) },
]);