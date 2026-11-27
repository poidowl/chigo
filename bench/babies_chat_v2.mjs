// babies_chat_v2.mjs — 2人のニューラルベビー（あか・あお）の50音会話実験
// 実行: node bench/babies_chat_v2.mjs
//
// 実験の流れ:
//   Phase 1 (別々の体験):
//     「あか」は森と川で「ひ・あつ」「あめ・きえ・ひ」を体験
//     「あお」は山で「くま・こわ・にげ」「しか・うま・くう」を体験
//   Phase 2 (赤ん坊どうしの50音会話):
//     お互いの50音ID列だけを投げ合いながら、相手の知識がシナプスに伝染し、
//     一語文 → 二語文へと発達していく様子を観察する。

import { encode50, decode50 } from '../src/codec50.mjs';
import { createNeuralBabyV2 } from '../src/neural_baby_v2.mjs';

const aka = createNeuralBabyV2({ name: 'あか', seed: 11 });
const ao = createNeuralBabyV2({ name: 'あお', seed: 29 });

console.log('=== Phase 1: それぞれ別々の縄文体験（初期知識ゼロから） ===');
const akaExperiences = [
  'ひ・あつ', 'ひ・もえ', 'あめ・ひ・きえ', 'さむ・ひ・たす',
  'ひ・あつ', 'あめ・さむ', 'あめ・ひ・きえ', 'みず・のむ',
];
const aoExperiences = [
  'くま・こわ', 'くま・にげ', 'しか・うま', 'はら・くう',
  'くま・こわ・にげ', 'しか・とる・くう', 'み・くう', 'よ・ねる',
];

for (const exp of akaExperiences) aka.experienceText(exp, 0.32);
for (const exp of aoExperiences) ao.experienceText(exp, 0.32);

console.log('体験直後のシナプス結合（まだお互いに知らないことがある）:');
console.table([
  {
    baby: 'あか',
    'ひ→あつ': aka.getWeight('ひ', 'あつ'),
    'あめ→きえ': aka.getWeight('あめ', 'きえ'),
    'くま→にげ(未体験)': aka.getWeight('くま', 'にげ'),
    'しか→くう(未体験)': aka.getWeight('しか', 'くう'),
  },
  {
    baby: 'あお',
    'ひ→あつ(未体験)': ao.getWeight('ひ', 'あつ'),
    'あめ→きえ(未体験)': ao.getWeight('あめ', 'きえ'),
    'くま→にげ': ao.getWeight('くま', 'にげ'),
    'しか→くう': ao.getWeight('しか', 'くう'),
  },
]);

console.log('\n=== Phase 2: ニューラルベビー同士の50音会話（環境刺激つき） ===');
// 時々、同じ縄文の空の下で共通の出来事が起きる
const worldEvents = {
  0: 'あめ・くる',
  5: 'くま・くる',
  10: 'さむ・よ',
  15: 'しか・くる',
  20: 'ほし・うえ',
  25: 'あめ・くま',
};

let currentMsgIds = encode50('あめ。', { appendEos: true });

for (let turn = 0; turn < 30; turn++) {
  if (worldEvents[turn]) {
    const ev = worldEvents[turn];
    console.log(`\n  [そらの出来事] ${ev}`);
    const evIds = encode50(ev, { appendEos: true });
    aka.perceive50(evIds, 0.25);
    ao.perceive50(evIds, 0.25);
    currentMsgIds = evIds;
  }

  const speaker = turn % 2 === 0 ? aka : ao;
  const res = speaker.listenAndReply50(currentMsgIds);
  console.log(
    `[T${String(turn + 1).padStart(2, '0')}][St${res.stage}][驚${res.surprise.toFixed(2)}] ${speaker.name}: ${res.reply}`
  );
  currentMsgIds = res.outIds;
}

console.log('\n=== Phase 3: 会話後のシナプス結合（会話だけで相手の知識が伝染したか？） ===');
console.table([
  {
    baby: 'あか',
    stage: aka.getStage(),
    'ひ→あつ': aka.getWeight('ひ', 'あつ'),
    'あめ→きえ': aka.getWeight('あめ', 'きえ'),
    'くま→にげ(伝染)': aka.getWeight('くま', 'にげ'),
    'しか→くう(伝染)': aka.getWeight('しか', 'くう'),
  },
  {
    baby: 'あお',
    stage: ao.getStage(),
    'ひ→あつ(伝染)': ao.getWeight('ひ', 'あつ'),
    'あめ→きえ(伝染)': ao.getWeight('あめ', 'きえ'),
    'くま→にげ': ao.getWeight('くま', 'にげ'),
    'しか→くう': ao.getWeight('しか', 'くう'),
  },
]);
