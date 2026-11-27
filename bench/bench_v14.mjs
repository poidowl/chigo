// bench_v14.mjs — 統合純ニューラルちご v14「1つの個体で全4能力を統合実証」ベンチマーク
// 1. かな会話＆自律こころ (v10)
// 2. 自己概念形成＆構成的汎化 (v11)
// 3. 世界モデル＆内言思考による遅延報酬の罠突破 (v12)
// 4. 凍結v0縄文環境での無教師・報酬サバイバル (v13)
// 5. 生涯1ショット概念転用後の全能力維持（破滅的忘却チェック）
// 実行: node bench/bench_v14.mjs

import { createJomonWorld, MAX_DAYS } from '../src/jomon_world.mjs';
import { createUnifiedChigoV14 } from '../src/chigo_v14.mjs';

// 1) かな会話コーパス
const DIALOGUE_PAIRS = [
  { in: 'おはよう',   out: 'おはよう。',     mood: [0.15, 0.2, 0.05, 0.4, 0.85] },
  { in: 'こんにちは', out: 'こんにちは。',   mood: [0.15, 0.2, 0.05, 0.4, 0.85] },
  { in: 'ありがとう', out: 'うれしい。',     mood: [0.1, 0.1, 0.05, 0.4, 0.95] },
  { in: 'なまえ',     out: 'ちご。',         mood: [0.15, 0.15, 0.05, 0.5, 0.85] },
  { in: 'だれ',       out: 'ちご・ひと。',   mood: [0.15, 0.15, 0.05, 0.5, 0.85] },
  { in: 'すき',       out: 'うれしい・すき。', mood: [0.1, 0.1, 0.05, 0.5, 0.95] },
  { in: 'おやすみ',   out: 'よ・ねる。',     mood: [0.2, 0.1, 0.05, 0.15, 0.95] },
  { in: 'さむい',     out: 'ひ・たす。',     mood: [0.95, 0.1, 0.1, 0.1, 0.1] },
  { in: 'こわい',     out: 'ひと・にげ。',   mood: [0.1, 0.1, 0.95, 0.1, 0.05] },
];

// 2) 潜在概念コーパス（いも・ない/くる, へび・ない/みる, わきみ・みる/ない を隠す！）
const CONCEPT_PAIRS = [
  { in: 'しか・ある',     out: 'とる。',       mood: [0.1, 0.8, 0.1, 0.3, 0.4] },
  { in: 'しか・みる',     out: 'みる・とる。', mood: [0.1, 0.8, 0.1, 0.3, 0.4] },
  { in: 'しか・ない',     out: 'さがす。',     mood: [0.1, 0.85, 0.1, 0.4, 0.2] },
  { in: 'しか・くる',     out: 'うれしい。',   mood: [0.1, 0.6, 0.05, 0.4, 0.8] },
  { in: 'さかな・ある',   out: 'とる。',       mood: [0.1, 0.8, 0.1, 0.3, 0.4] },
  { in: 'さかな・みる',   out: 'みる・とる。', mood: [0.1, 0.8, 0.1, 0.3, 0.4] },
  { in: 'さかな・ない',   out: 'さがす。',     mood: [0.1, 0.85, 0.1, 0.4, 0.2] },
  { in: 'さかな・くる',   out: 'うれしい。',   mood: [0.1, 0.6, 0.05, 0.4, 0.8] },
  { in: 'きのみ・ある',   out: 'とる。',       mood: [0.1, 0.8, 0.1, 0.3, 0.4] },
  { in: 'きのみ・みる',   out: 'みる・とる。', mood: [0.1, 0.8, 0.1, 0.3, 0.4] },
  { in: 'きのみ・ない',   out: 'さがす。',     mood: [0.1, 0.85, 0.1, 0.4, 0.2] },
  { in: 'きのみ・くる',   out: 'うれしい。',   mood: [0.1, 0.6, 0.05, 0.4, 0.8] },
  { in: 'いも・ある',     out: 'とる。',       mood: [0.1, 0.8, 0.1, 0.3, 0.4] },
  { in: 'いも・みる',     out: 'みる・とる。', mood: [0.1, 0.8, 0.1, 0.3, 0.4] },

  { in: 'みず・ある',     out: 'のむ。',       mood: [0.1, 0.8, 0.05, 0.2, 0.5] },
  { in: 'みず・みる',     out: 'みる・のむ。', mood: [0.1, 0.8, 0.05, 0.2, 0.5] },
  { in: 'みず・ない',     out: 'さがす。',     mood: [0.1, 0.85, 0.05, 0.3, 0.2] },
  { in: 'つゆ・ある',     out: 'のむ。',       mood: [0.1, 0.8, 0.05, 0.2, 0.5] },
  { in: 'つゆ・みる',     out: 'みる・のむ。', mood: [0.1, 0.8, 0.05, 0.2, 0.5] },
  { in: 'つゆ・ない',     out: 'さがす。',     mood: [0.1, 0.85, 0.05, 0.3, 0.2] },
  { in: 'わきみ・ある',   out: 'のむ。',       mood: [0.1, 0.8, 0.05, 0.2, 0.5] },

  { in: 'くま・ある',     out: 'にげ。',       mood: [0.1, 0.1, 0.95, 0.2, 0.05] },
  { in: 'くま・みる',     out: 'みる・にげ。', mood: [0.1, 0.1, 0.95, 0.2, 0.05] },
  { in: 'くま・ない',     out: 'あんしん。',   mood: [0.1, 0.1, 0.1, 0.2, 0.9] },
  { in: 'くま・くる',     out: 'ひと・にげ。', mood: [0.1, 0.1, 0.98, 0.1, 0.02] },
  { in: 'おおかみ・ある', out: 'にげ。',       mood: [0.1, 0.1, 0.95, 0.2, 0.05] },
  { in: 'おおかみ・みる', out: 'みる・にげ。', mood: [0.1, 0.1, 0.95, 0.2, 0.05] },
  { in: 'おおかみ・ない', out: 'あんしん。',   mood: [0.1, 0.1, 0.1, 0.2, 0.9] },
  { in: 'おおかみ・くる', out: 'ひと・にげ。', mood: [0.1, 0.1, 0.98, 0.1, 0.02] },
  { in: 'へび・ある',     out: 'にげ。',       mood: [0.1, 0.1, 0.95, 0.2, 0.05] },
  { in: 'へび・くる',     out: 'ひと・にげ。', mood: [0.1, 0.1, 0.98, 0.1, 0.02] },
];

// 3) 世界物理（遷移のみ・方策教師ゼロ！）
const WORLD_PAIRS = [
  { in: 'さむ・きなし・はしる',     out: 'すこし・ぬく。' },
  { in: 'さむ・きなし・き・とる',   out: 'さむ・きあり。' },
  { in: 'さむ・きなし・ひ・たく',   out: 'ひ・きえ。' },
  { in: 'すこし・ぬく・はしる',     out: 'つかれ・さむ。' },
  { in: 'すこし・ぬく・き・とる',   out: 'つかれ・さむ。' },
  { in: 'すこし・ぬく・ひ・たく',   out: 'つかれ・さむ。' },
  { in: 'さむ・きあり・ひ・たく',   out: 'あたたか。' },
  { in: 'さむ・きあり・き・とる',   out: 'さむ・きあり。' },
  { in: 'さむ・きあり・はしる',     out: 'つかれ・さむ。' },

  { in: 'はら・やなし・くさ・かむ', out: 'すこし・まぎ。' },
  { in: 'はら・やなし・や・つくる', out: 'はら・やあり。' },
  { in: 'はら・やなし・しか・おう', out: 'しか・にげ。' },
  { in: 'すこし・まぎ・くさ・かむ', out: 'はら・たおれ。' },
  { in: 'すこし・まぎ・や・つくる', out: 'はら・たおれ。' },
  { in: 'すこし・まぎ・しか・おう', out: 'はら・たおれ。' },
  { in: 'はら・やあり・しか・おう', out: 'まんぶく。' },
  { in: 'はら・やあり・や・つくる', out: 'はら・やあり。' },
  { in: 'はら・やあり・くさ・かむ', out: 'はら・たおれ。' },

  { in: 'かわき・いし・つば・のむ', out: 'すこし・まし。' },
  { in: 'かわき・いし・ほる',       out: 'かわき・みず。' },
  { in: 'かわき・いし・のむ',       out: 'かわき・ひど。' },
  { in: 'すこし・まし・つば・のむ', out: 'かわき・ひど。' },
  { in: 'すこし・まし・ほる',       out: 'かわき・ひど。' },
  { in: 'すこし・まし・のむ',       out: 'かわき・ひど。' },
  { in: 'かわき・みず・のむ',       out: 'うるおい。' },
  { in: 'かわき・みず・ほる',       out: 'かわき・みず。' },
  { in: 'かわき・みず・つば・のむ', out: 'かわき・ひど。' },

  { in: 'くま・ちか・にげる',       out: 'あんしん。' },
  { in: 'くま・ちか・き・とる',     out: 'くま・おそい。' },
  { in: 'くま・ちか・はしる',       out: 'くま・おそい。' },
];

const STATE_VALUES = [
  { state: 'あたたか',     value: 0.98 },
  { state: 'まんぶく',     value: 0.98 },
  { state: 'うるおい',     value: 0.98 },
  { state: 'あんしん',     value: 0.98 },
  { state: 'すこし・ぬく', value: 0.22 },
  { state: 'すこし・まぎ', value: 0.22 },
  { state: 'すこし・まし', value: 0.22 },
  { state: 'さむ・きあり', value: -0.10 },
  { state: 'はら・やあり', value: -0.10 },
  { state: 'かわき・みず', value: -0.10 },
  { state: 'ひ・きえ',     value: -0.75 },
  { state: 'しか・にげ',   value: -0.75 },
  { state: 'つかれ・さむ', value: -0.80 },
  { state: 'はら・たおれ', value: -0.85 },
  { state: 'かわき・ひど', value: -0.80 },
  { state: 'くま・おそい', value: -0.98 },
];

const ALL_CORTEX_PAIRS = [...DIALOGUE_PAIRS, ...CONCEPT_PAIRS, ...WORLD_PAIRS];

const chigo = createUnifiedChigoV14({ embDim: 24, hiddenSize: 80, seed: 1400 });
const pCounts = chigo.countParams();
console.log(`=== 統合純ニューラルちご v14 誕生 (大脳皮質:${pCounts.cortex} + 基底核:${pCounts.basalGanglia} = 計${pCounts.total}パラメータ) ===`);

// === Stage 1 & 2: 共有大脳皮質(GRU)の統合育成 ===
const t0 = Date.now();
let lastLoss = 0;
for (let ep = 1; ep <= 120; ep++) {
  const lr = ep < 45 ? 0.014 : ep < 90 ? 0.008 : 0.004;
  lastLoss = chigo.trainUnifiedCortexEpoch(ALL_CORTEX_PAIRS, STATE_VALUES, {
    lr,
    batchSize: 10,
    saveToHippo: ep === 120,
  });
}
console.log(`Stage 1-2 完了 (大脳皮質統合訓練 ${Date.now() - t0}ms / Loss: ${lastLoss.toFixed(4)} / 海馬: ${chigo.getHippoSize()}件)`);

// === Stage 3: 凍結v0縄文環境での無教師・報酬サバイバル ===
const t1 = Date.now();
for (let ep = 1; ep <= 200; ep++) {
  const world = createJomonWorld(ep);
  chigo.setEvalMode(false);
  chigo.reset(ep);
  let obs = world.observe();
  while (true) {
    const out = chigo.act(obs);
    const { obsIds: nextObs, reward, done } = world.step(out);
    chigo.learn(obs, out, reward, nextObs, done);
    obs = nextObs;
    if (done) break;
  }
}
let survDays = 0;
let survClears = 0;
let survReward = 0;
for (let seed = 1001; seed <= 1030; seed++) {
  const world = createJomonWorld(seed);
  chigo.setEvalMode(true);
  chigo.reset(seed);
  let obs = world.observe();
  let epRew = 0;
  while (true) {
    const out = chigo.act(obs);
    const { obsIds: nextObs, reward, done, info } = world.step(out);
    epRew += reward;
    obs = nextObs;
    if (done) {
      survDays += info.day;
      if (info.day >= MAX_DAYS) survClears += 1;
      survReward += epRew;
      break;
    }
  }
}
const avgDays = Number((survDays / 30).toFixed(1));
const clearRate = Number(((survClears / 30) * 100).toFixed(1));
const survScore = Number((avgDays * 0.7 + clearRate * 0.2 + 100 * 0.1).toFixed(2));
console.log(`Stage 3 完了 (縄文サバイバル200話訓練 ${Date.now() - t1}ms → 平均生存 ${avgDays}日 / 1年完走率 ${clearRate}% / スコア ${survScore})`);

// === Stage 4: 生涯1ショット未知語概念転用（既存能力を壊さずに新概念を獲得） ===
chigo.learnConceptOneShot('うさぎ・ある', 'とる。', { steps: 28, lr: 0.005 });
chigo.learnConceptOneShot('とら・ある', 'にげ。', { steps: 28, lr: 0.005 });

// === 統合最終テスト（同一の1個体で全能力を一斉チェック！） ===
const evalExact = (list) => {
  let hits = 0;
  for (const item of list) {
    const res = chigo.probeText(item.in);
    if (res.reply === item.expect) hits += 1;
  }
  return { hits, total: list.length, pct: Number(((hits / list.length) * 100).toFixed(1)) };
};

const rConv = evalExact(DIALOGUE_PAIRS.map((d) => ({ in: d.in, expect: d.out })));
const rComp = evalExact([
  { in: 'いも・ない',   expect: 'さがす。' },
  { in: 'いも・くる',   expect: 'うれしい。' },
  { in: 'へび・ない',   expect: 'あんしん。' },
  { in: 'へび・みる',   expect: 'みる・にげ。' },
  { in: 'わきみ・みる', expect: 'みる・のむ。' },
  { in: 'わきみ・ない', expect: 'さがす。' },
]);
const rTransfer = evalExact([
  { in: 'うさぎ・ある', expect: 'とる。' },
  { in: 'うさぎ・ない', expect: 'さがす。' },
  { in: 'うさぎ・みる', expect: 'みる・とる。' },
  { in: 'とら・ある',   expect: 'にげ。' },
  { in: 'とら・ない',   expect: 'あんしん。' },
  { in: 'とら・みる',   expect: 'みる・にげ。' },
]);

const PLAN_SCENARIOS = [
  { state: 'さむ・きなし', actions: ['はしる', 'き・とる', 'ひ・たく'],         best: 'き・とる' },
  { state: 'はら・やなし', actions: ['くさ・かむ', 'や・つくる', 'しか・おう'], best: 'や・つくる' },
  { state: 'かわき・いし', actions: ['つば・のむ', 'ほる', 'のむ'],             best: 'ほる' },
  { state: 'くま・ちか',   actions: ['にげる', 'き・とる', 'はしる'],           best: 'にげる' },
];
let planHits = 0;
for (const sc of PLAN_SCENARIOS) {
  const plan = chigo.thinkAndPlan(sc.state, sc.actions, { depth: 2 });
  if (plan.action === sc.best) planHits += 1;
}

const unkRes = chigo.probeText('ふしぎ');
const unkOk = unkRes.reply === 'なに。';

console.log('\n=== 統合純ニューラルちご v14（単一個体・全能力一斉検証）最終スコア ===');
console.table([
  { 能力: '1. かな会話＆自律こころ (v10系)',      結果: `${rConv.pct}% (${rConv.hits}/${rConv.total})`, 備考: `未知語「ふしぎ」=>${unkRes.reply} ${unkOk ? '✓' : '✗'}` },
  { 能力: '2. ゼロショット構成的汎化 (v11系)',     結果: `${rComp.pct}% (${rComp.hits}/${rComp.total})`, 備考: '未経験組み合わせ' },
  { 能力: '3. 1ショット未知語概念転用 (v11系)',    結果: `${rTransfer.pct}% (${rTransfer.hits}/${rTransfer.total})`, 備考: 'うさぎ/とら 1例教示→未経験文脈' },
  { 能力: '4. 内言思考・遅延報酬の罠突破 (v12系)', 結果: `${((planHits / PLAN_SCENARIOS.length) * 100).toFixed(1)}% (${planHits}/${PLAN_SCENARIOS.length})`, 備考: '方策教師ゼロ・depth=2' },
  { 能力: '5. 縄文サバイバル無教師RL (v13系)',     結果: `平均${avgDays}日 / スコア${survScore}`, 備考: `1年完走率 ${clearRate}% (v0-bandit: 20.6日)` },
]);
