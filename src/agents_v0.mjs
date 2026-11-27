// agents_v0.mjs — 差し替え可能なエージェントAPI & v0 ベースライン群
// すべてのエージェントは以下の共通契約を守る：
//   act(obsIds: Uint8Array) -> Uint8Array (値域 0..49)
//   learn(obsIds, outIds, reward, nextObsIds, done) -> void

import { VOCAB_SIZE, TOKEN_EOS, encode50 } from './codec50.mjs';
import { createRng } from './jomon_world.mjs';

// ---------------------------------------------------------------------------
// 1) v0-random: 下限ベースライン（完全ランダム50音列）
// ---------------------------------------------------------------------------
export function createRandomAgent() {
  let rng = createRng(42);
  return {
    name: 'v0-random',
    reset(seed = 42) {
      rng = createRng(seed + 1000);
    },
    act(_obsIds) {
      const len = 3 + Math.floor(rng() * 5);
      const out = new Uint8Array(len + 1);
      for (let i = 0; i < len; i++) {
        out[i] = 4 + Math.floor(rng() * (VOCAB_SIZE - 4));
      }
      out[len] = TOKEN_EOS;
      return out;
    },
    learn() {},
  };
}

// ---------------------------------------------------------------------------
// 2) v0-ngram-rl: ゼロ依存・純50音トークン列を報酬変調で学ぶ極小バイグラム方策
//    重みは Float32Array(50 * 50) + 文脈バイアスのみ。意味の辞書を一切持たない。
// ---------------------------------------------------------------------------
export function createNgramRlAgent() {
  let rng = createRng(7);
  // 遷移ロジット: prevToken -> nextToken (50 x 50)
  const trans = new Float32Array(VOCAB_SIZE * VOCAB_SIZE);
  // 入力50音の出現トークン -> 先頭出力トークン (50 x 50)
  const obsToFirst = new Float32Array(VOCAB_SIZE * VOCAB_SIZE);
  let baselineReward = 0;

  function sampleSoftmax(logits, temperature = 1.0) {
    let maxV = -Infinity;
    for (let i = 0; i < VOCAB_SIZE; i++) {
      if (logits[i] > maxV) maxV = logits[i];
    }
    let sum = 0;
    const probs = new Float32Array(VOCAB_SIZE);
    for (let i = 0; i < VOCAB_SIZE; i++) {
      const e = Math.exp((logits[i] - maxV) / temperature);
      probs[i] = e;
      sum += e;
    }
    let r = rng() * sum;
    for (let i = 0; i < VOCAB_SIZE; i++) {
      r -= probs[i];
      if (r <= 0) return i;
    }
    return VOCAB_SIZE - 1;
  }

  return {
    name: 'v0-ngram-rl',
    reset(seed = 7) {
      rng = createRng(seed + 2000);
    },
    act(obsIds) {
      const logits = new Float32Array(VOCAB_SIZE);
      for (let i = 0; i < obsIds.length; i++) {
        const oid = obsIds[i];
        const base = oid * VOCAB_SIZE;
        for (let k = 0; k < VOCAB_SIZE; k++) {
          logits[k] += obsToFirst[base + k] * 0.05;
        }
      }
      const out = [];
      let tok = sampleSoftmax(logits, 1.0);
      out.push(tok);
      for (let step = 0; step < 5; step++) {
        if (tok === TOKEN_EOS) break;
        const row = trans.subarray(tok * VOCAB_SIZE, (tok + 1) * VOCAB_SIZE);
        tok = sampleSoftmax(row, 1.0);
        out.push(tok);
      }
      if (out[out.length - 1] !== TOKEN_EOS) out.push(TOKEN_EOS);
      return Uint8Array.from(out);
    },
    learn(obsIds, outIds, reward) {
      const adv = reward - baselineReward;
      baselineReward = baselineReward * 0.98 + reward * 0.02;
      const lr = 0.05 * adv;
      if (outIds.length === 0) return;

      const first = outIds[0];
      for (let i = 0; i < obsIds.length; i++) {
        obsToFirst[obsIds[i] * VOCAB_SIZE + first] += lr * 0.1;
      }
      for (let i = 0; i + 1 < outIds.length; i++) {
        trans[outIds[i] * VOCAB_SIZE + outIds[i + 1]] += lr;
      }
    },
    serialize() {
      return { trans: Array.from(trans), obsToFirst: Array.from(obsToFirst) };
    },
  };
}

// ---------------------------------------------------------------------------
// 3) v0-bandit: 50音入力のバイグラム特徴(Float32Array)から
//    縄文語フレーズを選択・オンライン学習する文脈バンディット (v0目標基準)
// ---------------------------------------------------------------------------
const PHRASES = [
  'ひをたす。',
  'どんぐりひろう。',
  'さかなつる。',
  'しかかり。',
  'たべる。',
  'やすむ。',
].map((s) => encode50(s, { appendEos: true }));

export function createContextualBanditAgent() {
  const FEAT_DIM = 256;
  const NUM_ACTIONS = PHRASES.length;
  const weights = new Float32Array(NUM_ACTIONS * FEAT_DIM);
  let rng = createRng(99);
  let epsilon = 0.25;
  let lastActionIdx = 0;

  function extractFeatures(obsIds) {
    // 50音ID列のユニグラム＋バイグラムを固定次元へハッシュ射影（語彙知識ゼロ）
    const active = [];
    for (let i = 0; i < obsIds.length; i++) {
      const a = obsIds[i];
      active.push((a * 31) & (FEAT_DIM - 1));
      if (i + 1 < obsIds.length) {
        const b = obsIds[i + 1];
        active.push(((a * 53 + b * 97) >>> 0) & (FEAT_DIM - 1));
      }
    }
    return active;
  }

  function scoreAction(actionIdx, activeFeats) {
    const offset = actionIdx * FEAT_DIM;
    let s = 0;
    for (let i = 0; i < activeFeats.length; i++) {
      s += weights[offset + activeFeats[i]];
    }
    return s / Math.max(1, activeFeats.length);
  }

  return {
    name: 'v0-bandit',
    reset(seed = 99) {
      rng = createRng(seed + 3000);
    },
    setEvalMode(isEval) {
      epsilon = isEval ? 0.02 : 0.2;
    },
    act(obsIds) {
      const feats = extractFeatures(obsIds);
      let chosen = 0;
      if (rng() < epsilon) {
        chosen = Math.floor(rng() * NUM_ACTIONS);
      } else {
        let bestScore = -Infinity;
        for (let a = 0; a < NUM_ACTIONS; a++) {
          const sc = scoreAction(a, feats);
          if (sc > bestScore) {
            bestScore = sc;
            chosen = a;
          }
        }
      }
      lastActionIdx = chosen;
      return PHRASES[chosen];
    },
    learn(obsIds, _outIds, reward, nextObsIds, done) {
      const feats = extractFeatures(obsIds);
      const qCur = scoreAction(lastActionIdx, feats);

      let maxNext = 0;
      if (!done && nextObsIds) {
        const nextFeats = extractFeatures(nextObsIds);
        maxNext = -Infinity;
        for (let a = 0; a < NUM_ACTIONS; a++) {
          const sc = scoreAction(a, nextFeats);
          if (sc > maxNext) maxNext = sc;
        }
      }

      const gamma = 0.92;
      const tdError = reward + gamma * maxNext - qCur;
      const lr = 0.08 / Math.max(1, feats.length);
      const offset = lastActionIdx * FEAT_DIM;
      for (let i = 0; i < feats.length; i++) {
        weights[offset + feats[i]] += lr * tdError;
      }
    },
    serialize() {
      return { featDim: FEAT_DIM, weights: Array.from(weights) };
    },
  };
}
