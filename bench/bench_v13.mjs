// bench_v13.mjs — 純ニューラルちご v13「無教師・報酬サバイバル」検証
// v0 凍結ベンチマーク（固定30シード x 最大120日 / 訓練200話）で
// v0ベースラインと直接対決！
// 実行: node bench/bench_v13.mjs

import { createJomonWorld, MAX_DAYS } from '../src/jomon_world.mjs';
import { createRandomAgent, createContextualBanditAgent } from '../src/agents_v0.mjs';
import { createSurvivalChigoV13 } from '../src/chigo_v13.mjs';

const TRAIN_EPISODES = 200;
const EVAL_SEEDS = Array.from({ length: 30 }, (_, i) => 1001 + i);

function runEpisode(agent, seed, training, actionStats = null) {
  const world = createJomonWorld(seed);
  if (typeof agent.setEvalMode === 'function') agent.setEvalMode(!training);
  agent.reset(seed);

  let obsIds = world.observe();
  let totalReward = 0;
  let validActions = 0;
  let steps = 0;

  while (true) {
    const seasonBefore = world.getState().season;
    const outIds = agent.act(obsIds);
    const { obsIds: nextObsIds, reward, done, info } = world.step(outIds);
    if (training && typeof agent.learn === 'function') {
      agent.learn(obsIds, outIds, reward, nextObsIds, done);
    }
    if (actionStats && info.action) {
      if (!actionStats[seasonBefore]) actionStats[seasonBefore] = {};
      actionStats[seasonBefore][info.action] = (actionStats[seasonBefore][info.action] || 0) + 1;
    }
    totalReward += reward;
    if (info.action !== 'idle' && info.valid50) validActions += 1;
    steps += 1;
    obsIds = nextObsIds;
    if (done) {
      return {
        days: info.day,
        totalReward,
        validRate: steps > 0 ? validActions / steps : 0,
        survivedYear: info.day >= MAX_DAYS,
      };
    }
  }
}

function evaluateAgent(factory, trackCurve = false) {
  const agent = factory();
  const t0 = Date.now();
  const curve = [];

  for (let ep = 1; ep <= TRAIN_EPISODES; ep++) {
    const r = runEpisode(agent, ep, true);
    if (trackCurve && (ep === 1 || ep === 40 || ep === 80 || ep === 140 || ep === 200)) {
      curve.push({ ep, trainDays: r.days, trainReward: Number(r.totalReward.toFixed(1)) });
    }
  }
  const trainMs = Date.now() - t0;

  let sumDays = 0;
  let maxDays = 0;
  let sumReward = 0;
  let sumValid = 0;
  let yearClears = 0;
  const actionStats = trackCurve ? {} : null;

  for (const seed of EVAL_SEEDS) {
    const res = runEpisode(agent, seed, false, actionStats);
    sumDays += res.days;
    if (res.days > maxDays) maxDays = res.days;
    sumReward += res.totalReward;
    sumValid += res.validRate;
    if (res.survivedYear) yearClears += 1;
  }

  const n = EVAL_SEEDS.length;
  const avgDays = sumDays / n;
  const avgReward = sumReward / n;
  const avgValid = (sumValid / n) * 100;
  const clearRate = (yearClears / n) * 100;
  const score = avgDays * 0.7 + clearRate * 0.2 + avgValid * 0.1;

  return {
    summary: {
      name: agent.name,
      avgDays: Number(avgDays.toFixed(1)),
      maxDays,
      clearRate: Number(clearRate.toFixed(1)),
      validRate: Number(avgValid.toFixed(1)),
      avgReward: Number(avgReward.toFixed(2)),
      score: Number(score.toFixed(2)),
      trainMs,
    },
    curve,
    actionStats,
  };
}

const rRandom = evaluateAgent(createRandomAgent, false);
const rBandit = evaluateAgent(createContextualBanditAgent, false);
const rV13 = evaluateAgent(createSurvivalChigoV13, true);

console.log('=== 縄文サバイバル 50音AGI ベンチマーク (固定30シード x 最大120日 / 訓練200話) ===');
console.table([rRandom.summary, rBandit.summary, rV13.summary]);

console.log('\n=== ちご v13 発達曲線（生身で放り込まれてからの成長） ===');
console.table(rV13.curve);

console.log('\n=== ちご v13 が報酬だけから自力獲得した季節別行動分布（評価30シード合計） ===');
console.table(rV13.actionStats);
