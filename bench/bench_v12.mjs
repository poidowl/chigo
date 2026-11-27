// bench_v12.mjs — 純ニューラルちご v12「世界モデル＋内言思考ループ」検証
// 方策の正解ラベル（教師あり行動指示）はゼロ！
// 世界の物理（状態・行動→次状態）と身体価値（快/不快）だけから、
// 1手先反射 (depth=1) vs 2手先内言思考 (depth=2) で「遅延報酬の罠」対決を行う。
// 実行: node bench/bench_v12.mjs

import { createWorldModelChigoV12 } from '../src/chigo_v12.mjs';

// 50音の世界物理（遷移データのみ。どれが「正解の行動」かは一切教えない！）
const WORLD_TRANSITIONS = [
  // シナリオ1: さむ・きなし (罠: はしる は1手目+0.2だが2手目で疲れて-0.7 / 正解ルート: き・とる → ひ・たく → あたたか+1.0)
  { state: 'さむ・きなし',   action: 'はしる',     nextState: 'すこし・ぬく' },
  { state: 'さむ・きなし',   action: 'き・とる',   nextState: 'さむ・きあり' },
  { state: 'さむ・きなし',   action: 'ひ・たく',   nextState: 'ひ・きえ' },
  { state: 'すこし・ぬく',   action: 'はしる',     nextState: 'つかれ・さむ' },
  { state: 'すこし・ぬく',   action: 'き・とる',   nextState: 'つかれ・さむ' },
  { state: 'すこし・ぬく',   action: 'ひ・たく',   nextState: 'つかれ・さむ' },
  { state: 'さむ・きあり',   action: 'ひ・たく',   nextState: 'あたたか' },
  { state: 'さむ・きあり',   action: 'き・とる',   nextState: 'さむ・きあり' },
  { state: 'さむ・きあり',   action: 'はしる',     nextState: 'つかれ・さむ' },

  // シナリオ2: はら・やなし (罠: くさ・かむ は1手目+0.2→2手目-0.7 / 正解ルート: や・つくる → しか・おう → まんぷく+1.0)
  { state: 'はら・やなし',   action: 'くさ・かむ', nextState: 'すこし・まぎ' },
  { state: 'はら・やなし',   action: 'や・つくる', nextState: 'はら・やあり' },
  { state: 'はら・やなし',   action: 'しか・おう', nextState: 'しか・にげ' },
  { state: 'すこし・まぎ',   action: 'くさ・かむ', nextState: 'はら・たおれ' },
  { state: 'すこし・まぎ',   action: 'や・つくる', nextState: 'はら・たおれ' },
  { state: 'すこし・まぎ',   action: 'しか・おう', nextState: 'はら・たおれ' },
  { state: 'はら・やあり',   action: 'しか・おう', nextState: 'まんぷく' },
  { state: 'はら・やあり',   action: 'や・つくる', nextState: 'はら・やあり' },
  { state: 'はら・やあり',   action: 'くさ・かむ', nextState: 'はら・たおれ' },

  // シナリオ3: かわき・いし (罠: つば・のむ は1手目+0.2→2手目-0.7 / 正解ルート: ほる → のむ → うるおい+1.0)
  { state: 'かわき・いし',   action: 'つば・のむ', nextState: 'すこし・まし' },
  { state: 'かわき・いし',   action: 'ほる',       nextState: 'かわき・みず' },
  { state: 'かわき・いし',   action: 'のむ',       nextState: 'かわき・ひど' },
  { state: 'すこし・まし',   action: 'つば・のむ', nextState: 'かわき・ひど' },
  { state: 'すこし・まし',   action: 'ほる',       nextState: 'かわき・ひど' },
  { state: 'すこし・まし',   action: 'のむ',       nextState: 'かわき・ひど' },
  { state: 'かわき・みず',   action: 'のむ',       nextState: 'うるおい' },
  { state: 'かわき・みず',   action: 'ほる',       nextState: 'かわき・みず' },
  { state: 'かわき・みず',   action: 'つば・のむ', nextState: 'かわき・ひど' },

  // シナリオ4: よ・くら (罠: うずくまる は1手目+0.2→2手目-0.8 / 正解ルート: いし・うつ → き・くべる → あたたか+1.0)
  { state: 'よ・くら',       action: 'うずくまる', nextState: 'すこし・やす' },
  { state: 'よ・くら',       action: 'いし・うつ', nextState: 'よ・ひだね' },
  { state: 'よ・くら',       action: 'き・くべる', nextState: 'ひ・きえ' },
  { state: 'すこし・やす',   action: 'うずくまる', nextState: 'よ・こごえ' },
  { state: 'すこし・やす',   action: 'いし・うつ', nextState: 'よ・こごえ' },
  { state: 'すこし・やす',   action: 'き・くべる', nextState: 'よ・こごえ' },
  { state: 'よ・ひだね',     action: 'き・くべる', nextState: 'あたたか' },
  { state: 'よ・ひだね',     action: 'いし・うつ', nextState: 'よ・ひだね' },
  { state: 'よ・ひだね',     action: 'うずくまる', nextState: 'よ・こごえ' },

  // シナリオ5: くま・ちか (即時危険: 1手目で逃げるのが正解)
  { state: 'くま・ちか',     action: 'にげる',     nextState: 'あんしん' },
  { state: 'くま・ちか',     action: 'き・とる',   nextState: 'くま・おそい' },
  { state: 'くま・ちか',     action: 'はしる',     nextState: 'くま・おそい' },
];

// 身体が感じる快・不快（価値 V ∈ [-1, +1]）
const STATE_VALUES = [
  // ゴール（快: +1.0）
  { state: 'あたたか',     value: 0.98 },
  { state: 'まんぷく',     value: 0.98 },
  { state: 'うるおい',     value: 0.98 },
  { state: 'あんしん',     value: 0.98 },
  // 遅延報酬の罠（1手目だけ少し嬉しい: +0.22）
  { state: 'すこし・ぬく', value: 0.22 },
  { state: 'すこし・まぎ', value: 0.22 },
  { state: 'すこし・まし', value: 0.22 },
  { state: 'すこし・やす', value: 0.22 },
  // 準備状態（1手目は手間がかかる: -0.10）
  { state: 'さむ・きあり', value: -0.10 },
  { state: 'はら・やあり', value: -0.10 },
  { state: 'かわき・みず', value: -0.10 },
  { state: 'よ・ひだね',   value: -0.10 },
  // 失敗・危険（不快: -0.75 ~ -0.98）
  { state: 'ひ・きえ',     value: -0.75 },
  { state: 'しか・にげ',   value: -0.75 },
  { state: 'つかれ・さむ', value: -0.80 },
  { state: 'はら・たおれ', value: -0.85 },
  { state: 'かわき・ひど', value: -0.80 },
  { state: 'よ・こごえ',   value: -0.90 },
  { state: 'くま・おそい', value: -0.98 },
];

const chigo = createWorldModelChigoV12({ embDim: 24, hiddenSize: 64, seed: 122 });

const t0 = Date.now();
let lastLosses = { transLoss: 0, valLoss: 0 };
for (let ep = 1; ep <= 120; ep++) {
  const lr = ep < 45 ? 0.015 : ep < 90 ? 0.008 : 0.004;
  lastLosses = chigo.trainWorldEpoch(WORLD_TRANSITIONS, STATE_VALUES, { lr, batchSize: 8 });
}
const trainMs = Date.now() - t0;

// 部門A: 世界モデル予測精度
let worldHits = 0;
for (const tr of WORLD_TRANSITIONS) {
  const pred = chigo.imagineNextState(tr.state, tr.action);
  if (pred === tr.nextState) worldHits += 1;
}

console.log(
  `=== ちご v12 世界モデル訓練完了 (${chigo.countParams()} パラメータ / ${trainMs}ms / 遷移Loss:${lastLosses.transLoss.toFixed(4)} / 価値Loss:${lastLosses.valLoss.toFixed(4)}) ===`
);
console.log(`部門A: 世界物理の想像精度: ${((worldHits / WORLD_TRANSITIONS.length) * 100).toFixed(1)}% (${worldHits}/${WORLD_TRANSITIONS.length})`);

// 部門B: 遅延報酬の罠 対決（1手先反射 depth=1 vs 2手先内言思考 depth=2）
const SCENARIOS = [
  { name: '寒い(木なし)',     state: 'さむ・きなし', actions: ['はしる', 'き・とる', 'ひ・たく'],         bestFirst: 'き・とる' },
  { name: '腹へり(矢なし)',   state: 'はら・やなし', actions: ['くさ・かむ', 'や・つくる', 'しか・おう'], bestFirst: 'や・つくる' },
  { name: '喉かわき(石場)',   state: 'かわき・いし', actions: ['つば・のむ', 'ほる', 'のむ'],             bestFirst: 'ほる' },
  { name: '暗い夜(火種なし)', state: 'よ・くら',     actions: ['うずくまる', 'いし・うつ', 'き・くべる'], bestFirst: 'いし・うつ' },
  { name: '熊が近い(即時)',   state: 'くま・ちか',   actions: ['にげる', 'き・とる', 'はしる'],           bestFirst: 'にげる' },
];

console.log('\n=== 部門B: 遅延報酬の罠 対決（方策教師ゼロ！ 反射 depth=1 vs 内言思考 depth=2） ===');
let d1Hits = 0;
let d2Hits = 0;
for (const sc of SCENARIOS) {
  const p1 = chigo.thinkAndPlan(sc.state, sc.actions, { depth: 1 });
  const p2 = chigo.thinkAndPlan(sc.state, sc.actions, { depth: 2 });
  const ok1 = p1.action === sc.bestFirst;
  const ok2 = p2.action === sc.bestFirst;
  if (ok1) d1Hits += 1;
  if (ok2) d2Hits += 1;
  const thought = p2.trace
    .map((s) => `「${s.state}」→(${s.action})→「${s.imaginedNext}」[${s.value >= 0 ? '+' : ''}${s.value.toFixed(2)}]`)
    .join(' ⇒ ');
  console.log(`  [${sc.name}] 反射(d=1): ${p1.action.padEnd(6, '　')}${ok1 ? '✓' : '✗(罠!)'} | 内言(d=2): ${p2.action.padEnd(6, '　')}${ok2 ? '✓' : '✗'}`);
  console.log(`      内言トレース: ${thought}`);
}

// 部門C: 環境変化へのゼロショット再計画（方策を再訓練せず、世界物理だけ書き換えて計画が変わるか！）
console.log('\n=== 部門C: 環境変化へのゼロショット再計画（雨で火が消える世界へ変化） ===');
const UPDATED_TRANSITIONS = [
  ...WORLD_TRANSITIONS.filter((t) => !(t.state === 'さむ・きあり' && t.action === 'ひ・たく')),
  { state: 'さむ・きあり', action: 'ひ・たく',   nextState: 'ひ・きえ' },
  { state: 'さむ・きあり', action: 'ほら・いく', nextState: 'あんしん' },
  { state: 'さむ・きなし', action: 'ほら・いく', nextState: 'つかれ・さむ' },
];
for (let ep = 1; ep <= 28; ep++) {
  chigo.trainWorldEpoch(UPDATED_TRANSITIONS, STATE_VALUES, { lr: 0.006, batchSize: 8 });
}
const replan = chigo.thinkAndPlan('さむ・きあり', ['ひ・たく', 'ほら・いく', 'はしる'], { depth: 1 });
const replanOk = replan.action === 'ほら・いく';
console.log(
  `  雨で「ひ・たく→ひ・きえ」に変化後、「さむ・きあり」での自律選択 => ${replan.action} (想像: ${replan.trace[0].imaginedNext}, 価値: ${replan.score}) ${replanOk ? '✓' : '✗'}`
);

console.log('\n=== 純ニューラルちご v12（世界モデル＋内言思考）最終スコア ===');
console.table([
  {
    モード: '1手先反射 (depth=1 / 先読みなし)',
    世界物理予測: `${((worldHits / WORLD_TRANSITIONS.length) * 100).toFixed(1)}%`,
    遅延報酬の罠突破: `${((d1Hits / SCENARIOS.length) * 100).toFixed(1)}% (${d1Hits}/${SCENARIOS.length})`,
    環境変化ゼロショット再計画: '-',
  },
  {
    モード: '内言思考ループ (depth=2 / 50音ロールアウト)',
    世界物理予測: `${((worldHits / WORLD_TRANSITIONS.length) * 100).toFixed(1)}%`,
    遅延報酬の罠突破: `${((d2Hits / SCENARIOS.length) * 100).toFixed(1)}% (${d2Hits}/${SCENARIOS.length})`,
    環境変化ゼロショット再計画: replanOk ? '100% (OK)' : 'NG',
  },
]);
