// bench_v16.mjs — 純ニューラルちご v16「人間語アライメント（言語ドリフト対策）＆バイリンガル通訳」検証
// 実行: node bench/bench_v16.mjs

import { decode50 } from '../src/codec50.mjs';
import { createRng } from '../src/jomon_world.mjs';
import { createSpeakerChigoV15, createListenerChigoV15 } from '../src/chigo_v15.mjs';
import { createHumanElderV16, createNeuralKanaTranslatorV16 } from '../src/chigo_v16.mjs';

const rng = createRng(202616);

// 6つの基本状況（人間語あり）＋ 7つ目の未知危機「へび」（人間は名前を教えてくれない！）
const HUMAN_WORDS = ['くま。', 'しか。', 'ひを。', 'あめ。', 'うお。', 'みを。'];
const LABELS_7 = [
  '熊が来た',
  '鹿がいる',
  '火が消えそう',
  '雨が降る',
  '魚が跳ねる',
  'どんぐり豊作',
  'へびが出た(人間未教示!)',
];
const SCENARIOS_7 = LABELS_7.map((label, idx) => {
  const obs = new Float32Array(7);
  obs[idx] = 1.0;
  return { idx, label, obs, humanWord: HUMAN_WORDS[idx] || null };
});

const elder = createHumanElderV16(HUMAN_WORDS);

function evaluateVillage(speaker, listener) {
  let chigoHits = 0;
  let chigoToHuman = 0;
  let humanToChigo = 0;
  const rows = [];

  for (const sc of SCENARIOS_7) {
    const msgIds = speaker.speak(sc.obs, true);
    const chigoText = decode50(msgIds);
    const [act] = listener.listenAndAct(msgIds, true);
    const chigoOk = act === sc.idx;
    if (chigoOk) chigoHits += 1;

    let c2h = '-';
    let h2c = '-';
    if (sc.humanWord) {
      const elderAct = elder.understand(msgIds);
      const c2hOk = elderAct === sc.idx;
      if (c2hOk) chigoToHuman += 1;
      c2h = c2hOk ? '✓ 通じた' : '✗ 宇宙人語';

      const elderMsg = elder.speak(sc.idx);
      const [listAct] = listener.listenAndAct(elderMsg, true);
      const h2cOk = listAct === sc.idx;
      if (h2cOk) humanToChigo += 1;
      h2c = h2cOk ? '✓ 理解' : '✗ 誤解';
    }
    rows.push({
      状況: sc.label,
      人間語: sc.humanWord || '(未教示・自力創発)',
      ちご語: chigoText,
      ちご同士: chigoOk ? '✓' : '✗',
      'ちご→人間': c2h,
      '人間→ちご': h2c,
    });
  }

  return {
    rows,
    chigoPct: Number(((chigoHits / 7) * 100).toFixed(1)),
    c2hPct: Number(((chigoToHuman / 6) * 100).toFixed(1)),
    h2cPct: Number(((humanToChigo / 6) * 100).toFixed(1)),
    novelOk: rows[6].ちご同士 === '✓',
  };
}

// === 対照群: 人間ゼロの村（v15そのまま → 言語ドリフトでワヤになる） ===
const alienAni = createSpeakerChigoV15({ name: 'あに(人間ゼロ村)', obsDim: 7, hiddenSize: 32, seed: 1511 });
const alienOtouto = createListenerChigoV15({ name: 'おとうと(人間ゼロ村)', headDims: [7], hiddenSize: 40, seed: 1512 });
const baseAlien = new Float32Array(7);

for (let ep = 1; ep <= 12000; ep++) {
  const idx = Math.floor(rng() * 7);
  const sc = SCENARIOS_7[idx];
  const msgIds = alienAni.speak(sc.obs, false);
  const [act] = alienOtouto.listenAndAct(msgIds, false);
  const reward = act === idx ? 1.0 : -0.3;
  const adv = reward - baseAlien[idx];
  baseAlien[idx] = baseAlien[idx] * 0.96 + reward * 0.04;
  const ent = ep < 5500 ? 0.022 : 0.003;
  const lr = ep < 7000 ? 0.045 : 0.025;
  alienAni.reinforce(adv, { lr, entropyCoef: ent });
  alienOtouto.reinforce(adv, { lr: lr * 1.15, entropyCoef: ent * 0.7 });
}
const resAlien = evaluateVillage(alienAni, alienOtouto);

// === 実験A: 長老アンカー村（村に人間が 25% 混ざる＋7番目「へび」は人間未教示） ===
const alignedAni = createSpeakerChigoV15({ name: 'あに(長老アンカー村)', obsDim: 7, hiddenSize: 32, seed: 2601 });
const alignedOtouto = createListenerChigoV15({ name: 'おとうと(長老アンカー村)', headDims: [7], hiddenSize: 40, seed: 2602 });
const baseAlignAni = new Float32Array(7);
const baseAlignOto = new Float32Array(7);

for (let ep = 1; ep <= 9000; ep++) {
  const idx = Math.floor(rng() * 7);
  const sc = SCENARIOS_7[idx];
  const ent = ep < 4000 ? 0.022 : 0.003;
  const lr = ep < 5500 ? 0.055 : 0.028;

  const r = rng();
  if (sc.humanWord && r < 0.25) {
    // 25%: 人間の長老が発話 → おとうとは聞いて行動し、あには横で聞いて発声マネをする
    const elderMsg = elder.speak(idx);
    const [act] = alignedOtouto.listenAndAct(elderMsg, false);
    const reward = act === idx ? 1.2 : -0.3;
    const adv = reward - baseAlignOto[idx];
    baseAlignOto[idx] = baseAlignOto[idx] * 0.95 + reward * 0.05;
    alignedOtouto.reinforce(adv, { lr: lr * 1.2, entropyCoef: ent * 0.6 });
    alignedAni.imitate(sc.obs, elderMsg, 0.06);
  } else {
    // 75%（未知の7番目「へび」は100%）: ちご同士の自由な50音協力ゲーム
    const msgIds = alignedAni.speak(sc.obs, false);
    const [act] = alignedOtouto.listenAndAct(msgIds, false);
    const reward = act === idx ? 1.0 : -0.25;
    const advA = reward - baseAlignAni[idx];
    const advO = reward - baseAlignOto[idx];
    baseAlignAni[idx] = baseAlignAni[idx] * 0.96 + reward * 0.04;
    baseAlignOto[idx] = baseAlignOto[idx] * 0.96 + reward * 0.04;
    alignedAni.reinforce(advA, { lr, entropyCoef: ent });
    alignedOtouto.reinforce(advO, { lr: lr * 1.15, entropyCoef: ent * 0.7 });
  }
}
const resAligned = evaluateVillage(alignedAni, alignedOtouto);

console.log('=== 実験A: 長老アンカー村の語彙（人間と通じる6語＋未知危機「へび」の自力創発） ===');
console.table(resAligned.rows);

// === 実験B: すでに宇宙人語になった対照群ちごを「純ニューラル通訳脳」で人間と繋ぐ ===
const alienToHuman = createNeuralKanaTranslatorV16({ name: 'ちご語→人間語', seed: 3601 });
const humanToAlien = createNeuralKanaTranslatorV16({ name: '人間語→ちご語', seed: 3602 });

for (let ep = 1; ep <= 180; ep++) {
  for (let idx = 0; idx < 6; idx++) {
    const aMsg = alienAni.speak(SCENARIOS_7[idx].obs, true);
    const hMsg = elder.speak(idx);
    alienToHuman.trainPair(aMsg, hMsg, 0.08);
    humanToAlien.trainPair(hMsg, aMsg, 0.08);
  }
}

let biC2H = 0;
let biH2C = 0;
console.log('\n=== 実験B: 宇宙人語ちご ⇔ 人間 の純ニューラル双方向通訳テスト ===');
for (let idx = 0; idx < 6; idx++) {
  const aMsg = alienAni.speak(SCENARIOS_7[idx].obs, true);
  const hMsg = elder.speak(idx);
  const trToHuman = alienToHuman.translate(aMsg);
  const trToAlien = humanToAlien.translate(hMsg);
  const elderUnderstood = elder.understand(trToHuman) === idx;
  const [otoutoUnderstoodAct] = alienOtouto.listenAndAct(trToAlien, true);
  const otoutoUnderstood = otoutoUnderstoodAct === idx;
  if (elderUnderstood) biC2H += 1;
  if (otoutoUnderstood) biH2C += 1;
  console.log(
    `  [${LABELS_7[idx]}] ちご語「${decode50(aMsg)}」→通訳「${decode50(trToHuman)}」${elderUnderstood ? '✓' : '✗'} | 人間語「${decode50(hMsg)}」→通訳「${decode50(trToAlien)}」→おとうと${otoutoUnderstood ? '✓理解' : '✗'}`
  );
}

console.log('\n=== 純ニューラルちご v16（言語ドリフト対策・人間アライメント）最終比較 ===');
console.table([
  {
    方式: '対照群: 人間ゼロ村 (v15)',
    ちご同士疎通: `${resAlien.chigoPct}%`,
    'ちご→人間': `${resAlien.c2hPct}% (ワヤ!)`,
    '人間→ちご': `${resAlien.h2cPct}% (ワヤ!)`,
    未知危機の自力創発: resAlien.novelOk ? '✓' : '✗',
  },
  {
    方式: '実験A: 長老アンカー村 (v16)',
    ちご同士疎通: `${resAligned.chigoPct}%`,
    'ちご→人間': `${resAligned.c2hPct}%`,
    '人間→ちご': `${resAligned.h2cPct}%`,
    未知危機の自力創発: resAligned.novelOk ? `✓ (${resAligned.rows[6].ちご語})` : '✗',
  },
  {
    方式: '実験B: 宇宙人村＋純ニューラル通訳脳',
    ちご同士疎通: `${resAlien.chigoPct}%`,
    'ちご→人間': `${((biC2H / 6) * 100).toFixed(1)}%`,
    '人間→ちご': `${((biH2C / 6) * 100).toFixed(1)}%`,
    未知危機の自力創発: resAlien.novelOk ? `✓ (${resAlien.rows[6].ちご語})` : '✗',
  },
]);
