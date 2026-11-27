// listen_to_v1_parents.mjs — 「V1大人2体のえんえんヤマトコトバ会話」を横で聞き続けるニューラルベビー
// 実行: node bench/listen_to_v1_parents.mjs
//
// 構成:
//   - 話し手A (ちち): jomon_baby_v1 (Stage 2)
//   - 話し手B (はは): jomon_baby_v1 (Stage 2)
//   - 聞き役 (あかご): neural_baby_v2 (初期知識ゼロ・発話せずひたすら聴取してシナプスを育てる)

import { encode50, decode50 } from '../src/codec50.mjs';
import { createJomonBabyV1 } from '../src/jomon_baby_v1.mjs';
import { createNeuralBabyV2 } from '../src/neural_baby_v2.mjs';

const chichi = createJomonBabyV1({ startStage: 2 });
const haha = createJomonBabyV1({ startStage: 2 });
const baby = createNeuralBabyV2({ name: 'あかご', seed: 101 });

// 親2体の会話の種になる縄文の日常トピック（1〜2文字のヤマトコトバのみ）
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

// ベビーの成長チェック用プローブ（テスト時は学習率0で純粋な今の実力を見る）
const PROBES = ['あめ', 'くま', 'しか', 'さむ', 'はら'];

function probeBaby(label) {
  console.log(`\n--- [${label}] あかごテスト (St${baby.getStage()} / 聴取${baby.getSteps()}回) ---`);
  for (const p of PROBES) {
    // 学習率0で刺激だけ与えて返答を聞く
    const inIds = encode50(p, { appendEos: true });
    baby.perceive50(inIds, 0.0);
    const outIds = baby.speak50();
    console.log(`  よびかけ「${p}」 => あかご: ${decode50(outIds)}`);
  }
  console.log(
    `  シナプス: あめ→きえ=${baby.getWeight('あめ', 'きえ')}, くま→にげ=${baby.getWeight('くま', 'にげ')}, しか→とる=${baby.getWeight('しか', 'とる')}, さむ→ひ=${baby.getWeight('さむ', 'ひ')}, はら→くう=${baby.getWeight('はら', 'くう')}`
  );
}

// 0) まだ何も聞いていない真っ白な状態
probeBaby('聴取前・真っ白');

console.log('\n=== V1大人2体（ちち・はは）のヤマトコトバ会話（あかごは横で聴取） ===');

let lastUtteranceIds = encode50(TOPICS[0], { appendEos: true });
let prevText = '';
const TOTAL_TURNS = 80;

for (let turn = 0; turn < TOTAL_TURNS; turn++) {
  // 4ターンごと、または会話が同じ内容でループしたら新しい日常トピックを投入
  if (turn % 4 === 0) {
    const topic = TOPICS[Math.floor(turn / 4) % TOPICS.length];
    lastUtteranceIds = encode50(topic, { appendEos: true });
    // 話題のきっかけ自体もあかごの耳に入る
    baby.perceive50(lastUtteranceIds, 0.14);
    if (turn < 24) {
      console.log(`\n  [まわりの気配] ${topic}`);
    }
  }

  const speaker = turn % 2 === 0 ? chichi : haha;
  const speakerName = turn % 2 === 0 ? 'ちち(V1)' : 'はは(V1)';
  const outIds = speaker.think50(lastUtteranceIds);
  const outText = decode50(outIds);

  // あかごは横で黙って50音ID列を聞くだけ！
  const { surprise } = baby.perceive50(outIds, 0.14);

  // 最初の24ターンは親の会話ログをそのまま表示（以降は長くなるのでダイジェスト）
  if (turn < 24) {
    console.log(
      `[T${String(turn + 1).padStart(2, '0')}] ${speakerName}: ${outText.padEnd(14, '　')} (あかご驚:${surprise.toFixed(2)})`
    );
  }

  // もし完全に同じ返答が続いたら次の話題へ軽くずらす
  if (outText === prevText) {
    const fallback = TOPICS[(turn + 3) % TOPICS.length];
    lastUtteranceIds = encode50(fallback, { appendEos: true });
  } else {
    lastUtteranceIds = outIds;
  }
  prevText = outText;

  // 途中経過の発達チェック
  if (turn === 9) probeBaby('10ターン聴取後');
  if (turn === 29) probeBaby('30ターン聴取後');
}

// 最終チェック
probeBaby('80ターン聴取後');