// bench_v15.mjs — 純ニューラルちご v15「創発言語（50音ことばの自力発明）＆文化伝承」実験
// 初期語彙ゼロ・教師データゼロ・エージェント間の勾配共有ゼロ！
// 実行: node bench/bench_v15.mjs

import { createRng } from '../src/jomon_world.mjs';
import {
  createSpeakerChigoV15,
  createListenerChigoV15,
  inspectEmergentLexicon,
} from '../src/chigo_v15.mjs';
import { decode50 } from '../src/codec50.mjs';

const rng = createRng(202615);

// === 実験A: 6種の縄文危機・獲物に対する「50音ことばの自力発明」 ===
const JOMON_LABELS = [
  '熊が来た(にげる)',
  '鹿がいる(かる)',
  '火が消えそう(ひをたす)',
  '雨が降る(あまやどり)',
  '魚が跳ねる(つる)',
  'どんぐり豊作(ひろう)',
];
const SCENARIOS_A = JOMON_LABELS.map((label, idx) => {
  const obs = new Float32Array(6);
  obs[idx] = 1.0;
  return { label, obs, targetActs: [idx] };
});

const ani1 = createSpeakerChigoV15({ name: 'あに(第1世代)', obsDim: 6, hiddenSize: 32, seed: 1501 });
const otouto1 = createListenerChigoV15({ name: 'おとうと(第1世代)', headDims: [6], hiddenSize: 40, seed: 1502 });

const initLex = inspectEmergentLexicon(ani1, otouto1, SCENARIOS_A);
const initAcc = Number(((initLex.filter((x) => x.ok).length / SCENARIOS_A.length) * 100).toFixed(1));

const baselineA = new Float32Array(6);
const TOTAL_EP_A = 6000;
for (let ep = 1; ep <= TOTAL_EP_A; ep++) {
  const idx = Math.floor(rng() * SCENARIOS_A.length);
  const sc = SCENARIOS_A[idx];
  const msgIds = ani1.speak(sc.obs, false);
  const [act] = otouto1.listenAndAct(msgIds, false);
  const reward = act === idx ? 1.0 : -0.25;

  const adv = reward - baselineA[idx];
  baselineA[idx] = baselineA[idx] * 0.96 + reward * 0.04;

  const ent = ep < 2500 ? 0.018 : 0.003;
  const lr = ep < 3500 ? 0.055 : 0.03;
  ani1.reinforce(adv, { lr, entropyCoef: ent });
  otouto1.reinforce(adv, { lr: lr * 1.15, entropyCoef: ent * 0.7 });
}

const finalLexA = inspectEmergentLexicon(ani1, otouto1, SCENARIOS_A);
const finalAccA = Number(((finalLexA.filter((x) => x.ok).length / SCENARIOS_A.length) * 100).toFixed(1));
const uniqueWordsA = new Set(finalLexA.map((x) => x.発明した50音ことば)).size;

console.log('=== 実験A: ゼロから2体のちごが自力発明した「50音ことば（創発語彙）」 ===');
console.table(finalLexA.map(({ ok, ...rest }) => rest));
console.log(`  初期成功率: ${initAcc}% → 協力学習後: ${finalAccA}% (独立語彙数: ${uniqueWordsA}/6)`);

// === 実験B: 第1世代 → 生まれたての第2世代への「ことばの文化伝承」 ===
const otouto2 = createListenerChigoV15({ name: 'ちご(第2世代・初期知識ゼロ)', headDims: [6], hiddenSize: 40, seed: 2502 });
let gen2Baseline = 0;
let gen2ClearEp = null;
for (let ep = 1; ep <= 1200; ep++) {
  const idx = Math.floor(rng() * SCENARIOS_A.length);
  const sc = SCENARIOS_A[idx];
  // 第1世代のあには固定語彙で話し、第2世代だけが聞いて学ぶ！
  const msgIds = ani1.speak(sc.obs, true);
  const [act] = otouto2.listenAndAct(msgIds, false);
  const reward = act === idx ? 1.0 : -0.25;
  const adv = reward - gen2Baseline;
  gen2Baseline = gen2Baseline * 0.95 + reward * 0.05;
  otouto2.reinforce(adv, { lr: 0.07, entropyCoef: 0.005 });

  if (gen2ClearEp === null && ep % 50 === 0) {
    const check = inspectEmergentLexicon(ani1, otouto2, SCENARIOS_A);
    if (check.every((x) => x.ok)) gen2ClearEp = ep;
  }
}
const gen2Lex = inspectEmergentLexicon(ani1, otouto2, SCENARIOS_A);
const gen2Acc = Number(((gen2Lex.filter((x) => x.ok).length / SCENARIOS_A.length) * 100).toFixed(1));
console.log(`\n=== 実験B: 文化伝承（第1世代 → 生まれたて第2世代） ===`);
console.log(`  第2世代の理解率: ${gen2Acc}% (100%到達エピソード: ${gen2ClearEp ?? '未達'}話 / ゼロから発明は${TOTAL_EP_A}話)`);

// === 実験C: 構成的創発言語（3対象 × 3状態 = 9状況、うち1つを隠してゼロショット解読！） ===
const OBJS = ['くま', 'しか', 'さかな'];
const STATES = ['ちかい', 'とおい', 'おおい'];
const ALL_COMP = [];
for (let o = 0; o < 3; o++) {
  for (let s = 0; s < 3; s++) {
    const obs = new Float32Array(6);
    obs[o] = 1.0;
    obs[3 + s] = 1.0;
    ALL_COMP.push({ o, s, label: `${OBJS[o]}・${STATES[s]}`, obs, targetActs: [o, s] });
  }
}
// (しか, おおい) [o=1, s=2] を訓練から完全に隠す！
const TRAIN_COMP = ALL_COMP.filter((c) => !(c.o === 1 && c.s === 2));
const HELDOUT_COMP = ALL_COMP.find((c) => c.o === 1 && c.s === 2);

const aniC = createSpeakerChigoV15({ name: 'あに(構成的)', obsDim: 6, hiddenSize: 32, seed: 3501 });
const otoutoC = createListenerChigoV15({ name: 'おとうと(構成的)', headDims: [3, 3], hiddenSize: 40, seed: 3502 });
const baseC = new Float32Array(9);

for (let ep = 1; ep <= 16000; ep++) {
  const sc = TRAIN_COMP[Math.floor(rng() * TRAIN_COMP.length)];
  const key = sc.o * 3 + sc.s;
  const msgIds = aniC.speak(sc.obs, false);
  const [ao, as] = otoutoC.listenAndAct(msgIds, false);
  const exact = ao === sc.o && as === sc.s;
  const reward = exact ? 1.2 : ((ao === sc.o ? 0.2 : -0.25) + (as === sc.s ? 0.2 : -0.25));
  const adv = reward - baseC[key];
  baseC[key] = baseC[key] * 0.96 + reward * 0.04;

  const ent = ep < 6000 ? 0.022 : ep < 11000 ? 0.008 : 0.002;
  const lr = ep < 9000 ? 0.05 : 0.025;
  aniC.reinforce(adv, { lr, entropyCoef: ent });
  otoutoC.reinforce(adv, { lr: lr * 1.15, entropyCoef: ent * 0.7 });
}

const trainCompLex = inspectEmergentLexicon(aniC, otoutoC, TRAIN_COMP);
const trainCompAcc = Number(((trainCompLex.filter((x) => x.ok).length / TRAIN_COMP.length) * 100).toFixed(1));
const heldMsg = aniC.speak(HELDOUT_COMP.obs, true);
const [ho, hs] = otoutoC.listenAndAct(heldMsg, true);
const heldExact = ho === HELDOUT_COMP.o && hs === HELDOUT_COMP.s;
const heldPart = (ho === HELDOUT_COMP.o ? 1 : 0) + (hs === HELDOUT_COMP.s ? 1 : 0);

console.log('\n=== 実験C: 3対象×3状態の構成的創発言語（未経験「しか・おおい」ゼロショット解読） ===');
console.table(trainCompLex.map(({ ok, ...rest }) => rest));
console.log(
  `  未経験「${HELDOUT_COMP.label}」→ あにの初発話:「${decode50(heldMsg)}」→ おとうとの解読:「${OBJS[ho]}・${STATES[hs]}」 (属性一致 ${heldPart}/2) ${heldExact ? '✓ ゼロショット完全解読!' : ''}`
);

console.log('\n=== 純ニューラルちご v15（創発言語・社会知能）最終まとめ ===');
console.table([
  { 実験: 'A: ゼロから50音ことば自力発明 (6状況)', 結果: `${initAcc}% → ${finalAccA}% (${uniqueWordsA}語彙創発)` },
  { 実験: 'B: 第1世代→第2世代への文化伝承',        結果: `${gen2Acc}% (わずか${gen2ClearEp ?? '-'}話で習得)` },
  { 実験: 'C: 3×3構成的創発言語 (未経験解読)',     結果: `訓練8組:${trainCompAcc}% / 未経験ゼロショット:${heldPart}/2 (${heldExact ? '完全一致' : '部分一致'})` },
]);
