// bench.mjs — 縄文50音AGI 共通ベンチマークハーネス (v0)
// 実行: node bench/bench_v0.mjs
// 新バージョン (v1, v2, ...) を追加したら AGENTS 配列に足して同じシードで比較する。

import { createJomonWorld, MAX_DAYS } from '../src/jomon_world.mjs';
import {
  createRandomAgent,
  createNgramRlAgent,
  createContextualBanditAgent,
} from '../src/agents_v0.mjs';

const TRAIN_EPISODES = 200;
const EVAL_SEEDS = Array.from({ length: 30 }, (_, i) => 1001 + i);

function runEpisode(agent, seed, training) {
  const world = createJomonWorld(seed);
  if (typeof agent.setEvalMode === 'function') {
    agent.setEvalMode(!training);
  }
  agent.reset(seed);

  let obsIds = world.observe();
  let totalReward = 0;
  let validActions = 0;
  let steps = 0;

  while (true) {
    const outIds = agent.act(obsIds);
    const { obsIds: nextObsIds, reward, done, info } = world.step(outIds);
    if (training && typeof agent.learn === 'function') {
      agent.learn(obsIds, outIds, reward, nextObsIds, done);
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

function evaluateAgent(factory) {
  const agent = factory();

  // オンライン学習フェーズ（訓練シード 1..TRAIN_EPISODES）
  for (let ep = 1; ep <= TRAIN_EPISODES; ep++) {
    runEpisode(agent, ep, true);
  }

  // 評価フェーズ（固定の評価シード 30本）
  let sumDays = 0;
  let maxDays = 0;
  let sumReward = 0;
  let sumValid = 0;
  let yearClears = 0;

  for (const seed of EVAL_SEEDS) {
    const res = runEpisode(agent, seed, false);
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
  // 総合スコア: 生存日数(最大120) + 1年完走ボーナス + 有効発話率
  const score = avgDays * 0.7 + clearRate * 0.2 + avgValid * 0.1;

  return {
    name: agent.name,
    avgDays: Number(avgDays.toFixed(1)),
    maxDays,
    clearRate: Number(clearRate.toFixed(1)),
    validRate: Number(avgValid.toFixed(1)),
    avgReward: Number(avgReward.toFixed(2)),
    score: Number(score.toFixed(2)),
  };
}

const AGENTS = [
  createRandomAgent,
  createNgramRlAgent,
  createContextualBanditAgent,
];

const results = AGENTS.map(evaluateAgent);

console.log('=== 縄文50音AGI ベンチマーク (固定30シード x 最大120日 / 訓練200話) ===');
console.table(results);