// bench_chigo.mjs — 「ちごが ちち と はは の はなし を きく」世代対決ベンチマーク (v2 vs v3)
// 実行: node bench/bench_v3.mjs
//
// 同一の V1親2体（ちち・はは）のヤマトコトバ会話ストリームを
// 10ターン / 80ターン / 300ターン 聴かせ続け、v2 と v3 の成長・耐飽和性を比較する。

import { encode50, decode50 } from '../src/codec50.mjs';
import { createJomonBabyV1 } from '../src/jomon_baby_v1.mjs';
import { createNeuralBabyV2 } from '../src/neural_baby_v2.mjs';
import { createChigoV3 } from '../src/chigo_v3.mjs';

const TOPICS = [
  'あめ・くる',
  'くま・くる',
  'しか・ある',
  'さむ',
  'はら・へる',
  'みず',
  'よ',
  'うお',
  'み',
  'ひ',
];

// 評価プローブと期待される縄文因果（親の会話から自力で盗み聞きすべき対応）
const EVAL_PROBES = [
  { in: 'あめ', expect: ['きえ', 'ひ', 'さむ', 'みず'] },
  { in: 'くま', expect: ['こわ', 'にげ'] },
  { in: 'しか', expect: ['うま', 'とる', 'くう'] },
  { in: 'さむ', expect: ['ひ', 'たす'] },
  { in: 'はら', expect: ['くう', 'み'] },
  { in: 'みず', expect: ['のむ', 'かわ'] },
  { in: 'よ', expect: ['ねる', 'ひ'] },
];

/**
 * 同一の「ちち(V1) & はは(V1)」会話ストリーム（50音ID列の配列）を事前生成する。
 */
function generateParentConversation(turns) {
  const chichi = createJomonBabyV1({ startStage: 2 });
  const haha = createJomonBabyV1({ startStage: 2 });
  const stream = [];

  let lastIds = encode50(TOPICS[0], { appendEos: true });
  let prevText = '';

  for (let t = 0; t < turns; t++) {
    if (t % 4 === 0) {
      const topic = TOPICS[Math.floor(t / 4) % TOPICS.length];
      const topicIds = encode50(topic, { appendEos: true });
      stream.push({ speaker: 'そら', ids: topicIds, text: topic });
      lastIds = topicIds;
    }
    const speaker = t % 2 === 0 ? chichi : haha;
    const outIds = speaker.think50(lastIds);
    const outText = decode50(outIds);
    stream.push({
      speaker: t % 2 === 0 ? 'ちち' : 'はは',
      ids: outIds,
      text: outText,
    });

    if (outText === prevText) {
      lastIds = encode50(TOPICS[(t + 3) % TOPICS.length], { appendEos: true });
    } else {
      lastIds = outIds;
    }
    prevText = outText;
  }
  return stream;
}

function evalChigo(chigo, isV3) {
  let hits = 0;
  const answers = {};
  for (const p of EVAL_PROBES) {
    let reply = '';
    if (isV3) {
      reply = chigo.probeText(p.in);
    } else {
      chigo.perceive50(encode50(p.in, { appendEos: true }), 0.0);
      reply = decode50(chigo.speak50());
    }
    const ok = p.expect.some((kw) => reply.includes(kw));
    if (ok) hits += 1;
    answers[p.in] = `${reply}${ok ? '✓' : '✗'}`;
  }
  return {
    accuracy: Number(((hits / EVAL_PROBES.length) * 100).toFixed(1)),
    hits: `${hits}/${EVAL_PROBES.length}`,
    answers,
  };
}

const checkpoints = [10, 80, 300];
const summary = [];

for (const turns of checkpoints) {
  const stream = generateParentConversation(turns);

  const v2 = createNeuralBabyV2({ name: 'ちごv2(旧)', seed: 101 });
  const v3 = createChigoV3({ name: 'ちごv3(新)', seed: 101 });

  for (const ev of stream) {
    v2.perceive50(ev.ids, 0.14);
    v3.perceive50(ev.ids, 0.22);
  }

  const r2 = evalChigo(v2, false);
  const r3 = evalChigo(v3, true);

  summary.push({
    turns,
    'v2正答率': `${r2.accuracy}% (${r2.hits})`,
    'v3正答率': `${r3.accuracy}% (${r3.hits})`,
    'v3シナプス密度': `${v3.getDensity()}%`,
  });

  if (turns === 300) {
    console.log('=== 300ターン聴取後の発話比較（ちち・ははの会話をずっと聞いた後） ===');
    for (const p of EVAL_PROBES) {
      console.log(
        `  よびかけ「${p.in.padEnd(2, '　')}」 | v2: ${r2.answers[p.in].padEnd(16, '　')} | v3: ${r3.answers[p.in]}`
      );
    }
  }
}

console.log('\n=== 世代対決スコア（ちごv2 vs ちごv3） ===');
console.table(summary);