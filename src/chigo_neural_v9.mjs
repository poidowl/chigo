// chigo_neural_v9.mjs — 純ニューラルちご v9 (補完的学習系: 高速シナプス W_fast + 海馬リプレイ定着)
// 100%純ニューラル: 単語辞書・Map・DP分節・if文による定型返答は一切ゼロ！
//
// v8からの大進化（破滅的忘却なしのオンライン未知語習得）:
//   1. デュアル読み出しシナプス:
//      - W_out / W_ear       (長期皮質シナプス: ゆっくり学び、壊れにくい)
//      - W_fast / W_ear_fast (高速シナプス: 高次元リザバー状態 h のみに局所結合し、
//                             バイアスやこころへ干渉させずに未知語を即座に刻む)
//   2. 海馬エピソードバッファ (Uint8Arrayの50音列 + mood):
//      高驚きの直後に教わったエピソードを海馬へ一時保存し、
//      「ねむり (sleepConsolidate)」時に既存記憶と混ぜてリプレイ（Interleaved Replay）することで、
//      W_fast を完全 0 消去した後も長期皮質 W_out に定着させ、破滅的忘却をゼロに抑える。

import { VOCAB_SIZE, TOKEN_EOS, encode50, decode50 } from './codec50.mjs';
import { createRng } from './jomon_world.mjs';

const NUM_EMO = 5;
const IN_DIM = VOCAB_SIZE + NUM_EMO + 1; // 50音 + こころ5軸 + 驚き1軸 = 56
const IDX_SURPRISE = VOCAB_SIZE + NUM_EMO;

export function createPureNeuralChigoV9(opts = {}) {
  const { name = 'ちごv9(高速シナプス+海馬)', hiddenSize = 240, seed = 88 } = opts;
  const rng = createRng(seed);
  const H = hiddenSize;
  const FEAT_DIM = H + IN_DIM + 1;
  const EAR_DIM = H + VOCAB_SIZE + 1;

  // 入力結合 W_in
  const W_in = new Float32Array(H * IN_DIM);
  for (let i = 0; i < W_in.length; i++) {
    W_in[i] = (rng() * 2 - 1) * 0.5;
  }

  // スパース再帰結合 W_rec
  const recIndices = [];
  const recWeights = [];
  for (let i = 0; i < H; i++) {
    const idxs = [];
    const wts = [];
    for (let j = 0; j < H; j++) {
      if (rng() < 0.12) {
        idxs.push(j);
        wts.push((rng() * 2 - 1) * 0.2);
      }
    }
    recIndices.push(Int32Array.from(idxs));
    recWeights.push(Float32Array.from(wts));
  }

  // 3時間スケール漏れ率
  const alpha = new Float32Array(H);
  const third = Math.floor(H / 3);
  for (let i = 0; i < H; i++) {
    if (i < third) alpha[i] = 0.78 + rng() * 0.18;
    else if (i < third * 2) alpha[i] = 0.30 + rng() * 0.18;
    else alpha[i] = 0.08 + rng() * 0.08;
  }

  // 長期皮質シナプス (slow cortex)
  const W_out = new Float32Array(VOCAB_SIZE * FEAT_DIM);
  const V_out = new Float32Array(VOCAB_SIZE * FEAT_DIM);
  const W_ear = new Float32Array(VOCAB_SIZE * EAR_DIM);

  // 高速シナプス (fast weights: 240次元リザバー状態 h のみに接続して既存記憶への干渉を防ぐ)
  const W_fast = new Float32Array(VOCAB_SIZE * H);
  const W_ear_fast = new Float32Array(VOCAB_SIZE * H);

  // 海馬エピソード記憶バッファ（生の50音ID列と心の状態のみ）
  const hippocampus = []; // { inIds, outIds, mood, isNew }
  const MAX_HIPPO = 48;

  let pendingNovelIds = null;
  let pendingNovelMood = null;

  const h = new Float32Array(H);
  const hNext = new Float32Array(H);
  const feat = new Float32Array(FEAT_DIM);
  const earFeat = new Float32Array(EAR_DIM);
  const logits = new Float32Array(VOCAB_SIZE);
  const probs = new Float32Array(VOCAB_SIZE);
  const earProbs = new Float32Array(VOCAB_SIZE);
  const kokoro = new Float32Array([0.2, 0.2, 0.1, 0.2, 0.5]);

  let currentSurprise = 0.0;

  function resetState() {
    h.fill(0);
    currentSurprise = 0.0;
  }

  function setKokoro(arr) {
    for (let i = 0; i < NUM_EMO; i++) {
      kokoro[i] = Math.max(0, Math.min(1, arr[i] ?? 0));
    }
  }

  function recordHippo(inIds, outIds, mood, isNew = false) {
    const key = Array.from(inIds).join(',');
    const existingIdx = hippocampus.findIndex(
      (e) =>
        Array.from(e.inIds).join(',') === key &&
        Math.abs(e.mood[0] - mood[0]) < 0.2 &&
        Math.abs(e.mood[1] - mood[1]) < 0.2
    );
    const entry = {
      inIds: Uint8Array.from(inIds),
      outIds: Uint8Array.from(outIds),
      mood: Float32Array.from(mood),
      isNew,
    };
    if (existingIdx >= 0) {
      hippocampus[existingIdx] = entry;
    } else {
      hippocampus.push(entry);
      if (hippocampus.length > MAX_HIPPO) hippocampus.shift();
    }
  }

  function stepReservoir(tokenId, surpriseVal = currentSurprise) {
    for (let i = 0; i < H; i++) {
      const inRow = i * IN_DIM;
      let sum = W_in[inRow + tokenId] + W_in[inRow + IDX_SURPRISE] * surpriseVal;
      for (let e = 0; e < NUM_EMO; e++) {
        sum += W_in[inRow + VOCAB_SIZE + e] * kokoro[e];
      }
      const idxs = recIndices[i];
      const wts = recWeights[i];
      for (let k = 0; k < idxs.length; k++) {
        sum += h[idxs[k]] * wts[k];
      }
      const a = alpha[i];
      hNext[i] = (1 - a) * h[i] + a * Math.tanh(sum);
    }
    h.set(hNext);

    feat.fill(0);
    feat.set(h, 0);
    feat[H + tokenId] = 1.0;
    for (let e = 0; e < NUM_EMO; e++) feat[H + VOCAB_SIZE + e] = kokoro[e];
    feat[H + IDX_SURPRISE] = surpriseVal;
    feat[FEAT_DIM - 1] = 1.0;

    earFeat.fill(0);
    earFeat.set(h, 0);
    earFeat[H + tokenId] = 1.0;
    earFeat[EAR_DIM - 1] = 1.0;
  }

  function forwardEar(useFast = true) {
    let maxL = -Infinity;
    for (let v = 0; v < VOCAB_SIZE; v++) {
      const row = v * EAR_DIM;
      let s = 0;
      for (let k = 0; k < EAR_DIM; k++) {
        const fk = earFeat[k];
        if (fk !== 0) s += W_ear[row + k] * fk;
      }
      if (useFast) {
        const fRow = v * H;
        for (let k = 0; k < H; k++) s += W_ear_fast[fRow + k] * h[k];
      }
      logits[v] = s;
      if (s > maxL) maxL = s;
    }
    let sumExp = 0;
    for (let v = 0; v < VOCAB_SIZE; v++) {
      const e = Math.exp(logits[v] - maxL);
      earProbs[v] = e;
      sumExp += e;
    }
    for (let v = 0; v < VOCAB_SIZE; v++) earProbs[v] /= sumExp;
  }

  function forwardReadout(useFast = true) {
    let maxL = -Infinity;
    for (let v = 0; v < VOCAB_SIZE; v++) {
      const row = v * FEAT_DIM;
      let s = 0;
      for (let k = 0; k < FEAT_DIM; k++) {
        const fk = feat[k];
        if (fk !== 0) s += W_out[row + k] * fk;
      }
      if (useFast) {
        const fRow = v * H;
        for (let k = 0; k < H; k++) s += W_fast[fRow + k] * h[k];
      }
      logits[v] = s;
      if (s > maxL) maxL = s;
    }
    let sumExp = 0;
    for (let v = 0; v < VOCAB_SIZE; v++) {
      const e = Math.exp(logits[v] - maxL);
      probs[v] = e;
      sumExp += e;
    }
    for (let v = 0; v < VOCAB_SIZE; v++) probs[v] /= sumExp;
  }

  function trainCortexPair50(inIds, outIds, opts = {}) {
    const { lr = 0.035, isUnknownEpisode = false, saveToHippo = true } = opts;
    resetState();
    const momentum = 0.55;
    const wd = 0.0002;
    const surpriseLevel = isUnknownEpisode ? 1.0 : 0.0;

    if (!isUnknownEpisode && saveToHippo) {
      recordHippo(inIds, outIds, kokoro, false);
    }

    for (let t = 0; t < inIds.length; t++) {
      stepReservoir(inIds[t], surpriseLevel);
      if (!isUnknownEpisode && t + 1 < inIds.length) {
        const nextTok = inIds[t + 1];
        forwardEar(false);
        for (let v = 0; v < VOCAB_SIZE; v++) {
          const err = earProbs[v] - (v === nextTok ? 1.0 : 0.0);
          if (Math.abs(err) < 1e-4) continue;
          const row = v * EAR_DIM;
          for (let k = 0; k < EAR_DIM; k++) {
            const fk = earFeat[k];
            if (fk !== 0) W_ear[row + k] -= lr * 0.9 * (err * fk + wd * W_ear[row + k]);
          }
        }
      }
    }

    let loss = 0;
    for (let t = 0; t < outIds.length; t++) {
      const targetTok = outIds[t];
      forwardReadout(false);
      loss += -Math.log(Math.max(1e-7, probs[targetTok]));

      for (let v = 0; v < VOCAB_SIZE; v++) {
        const err = probs[v] - (v === targetTok ? 1.0 : 0.0);
        if (Math.abs(err) < 1e-4) continue;
        const row = v * FEAT_DIM;
        for (let k = 0; k < FEAT_DIM; k++) {
          const fk = feat[k];
          if (fk === 0) continue;
          const idx = row + k;
          const g = err * fk + wd * W_out[idx];
          V_out[idx] = momentum * V_out[idx] - lr * g;
          W_out[idx] += V_out[idx];
        }
      }
      if (t + 1 < outIds.length) stepReservoir(targetTok, surpriseLevel * 0.8);
    }
    return outIds.length > 0 ? loss / outIds.length : 0;
  }

  /**
   * 昼のオンライン即席記憶:
   * 高速シナプス W_fast / W_ear_fast（リザバー状態 h のみに結合）だけを更新。
   * バイアスやこころ次元に書き込まない＋昼間の新規エピソード同士を軽く復唱することで、
   * 既存皮質記憶 W_out を一切壊さずに複数の新未知語を同時に保持できる！
   */
  function imprintFast50(inIds, outIds, mood, stepsCount = 14) {
    recordHippo(inIds, outIds, mood, true);
    const daytimeNew = hippocampus.filter((e) => e.isNew);
    const anchorSample = hippocampus.filter((e) => !e.isNew).slice(0, 6);

    const lr = 0.075;
    const wd = 0.008;
    for (let rep = 0; rep < stepsCount; rep++) {
      // 昼間に新しく覚えた未知語＋既存アンカー数件に対して高速シナプスを整合調整
      const batch = [...daytimeNew, ...anchorSample];
      for (const item of batch) {
        setKokoro(item.mood);
        resetState();
        const isNewItem = item.isNew;

        for (let t = 0; t < item.inIds.length; t++) {
          stepReservoir(item.inIds[t], 0.0);
          if (isNewItem && t + 1 < item.inIds.length) {
            const nextTok = item.inIds[t + 1];
            forwardEar(true);
            for (let v = 0; v < VOCAB_SIZE; v++) {
              const err = earProbs[v] - (v === nextTok ? 1.0 : 0.0);
              if (Math.abs(err) < 1e-4) continue;
              const fRow = v * H;
              for (let k = 0; k < H; k++) {
                W_ear_fast[fRow + k] -= lr * (err * h[k] + wd * W_ear_fast[fRow + k]);
              }
            }
          }
        }

        const itemLr = isNewItem ? lr : lr * 0.45;
        for (let t = 0; t < item.outIds.length; t++) {
          const targetTok = item.outIds[t];
          forwardReadout(true);
          for (let v = 0; v < VOCAB_SIZE; v++) {
            const err = probs[v] - (v === targetTok ? 1.0 : 0.0);
            if (Math.abs(err) < 1e-4) continue;
            const fRow = v * H;
            for (let k = 0; k < H; k++) {
              W_fast[fRow + k] -= itemLr * (err * h[k] + wd * W_fast[fRow + k]);
            }
          }
          if (t + 1 < item.outIds.length) stepReservoir(targetTok, 0.0);
        }
      }
    }
  }

  /**
   * 夜の睡眠リプレイ (Sleep Consolidation):
   * 海馬にある新エピソード＋既存エピソードを交互リプレイして長期皮質 W_out / W_ear に定着させ、
   * 高速シナプス W_fast / W_ear_fast を完全クリア（0.0）する！
   */
  function sleepConsolidate(epochs = 24) {
    const NANI_IDS = encode50('なに。', { appendEos: true });
    for (let ep = 0; ep < epochs; ep++) {
      const lr = ep < 12 ? 0.032 : 0.018;
      for (const e of hippocampus) {
        setKokoro(e.mood);
        const repeats = e.isNew ? 2 : 1;
        for (let r = 0; r < repeats; r++) {
          trainCortexPair50(e.inIds, e.outIds, { lr, isUnknownEpisode: false, saveToHippo: false });
        }
      }
      const noise = new Uint8Array([4 + Math.floor(rng() * 46), 4 + Math.floor(rng() * 46), TOKEN_EOS]);
      setKokoro([0.2, 0.2, 0.1, 0.8, 0.3]);
      trainCortexPair50(noise, NANI_IDS, { lr: lr * 0.7, isUnknownEpisode: true, saveToHippo: false });
    }

    W_fast.fill(0);
    W_ear_fast.fill(0);
    for (const e of hippocampus) e.isNew = false;
  }

  function interact50(promptIds, opts = {}) {
    const { maxLen = 14, mood = null, allowOnlineLearn = true } = opts;
    if (mood) setKokoro(mood);

    if (allowOnlineLearn && pendingNovelIds) {
      const targetIn = pendingNovelIds;
      const targetMood = pendingNovelMood;
      pendingNovelIds = null;
      pendingNovelMood = null;
      imprintFast50(targetIn, promptIds, targetMood, 14);
      return respond50(targetIn, { maxLen, mood: targetMood, recordPending: false });
    }

    return respond50(promptIds, { maxLen, mood, recordPending: allowOnlineLearn });
  }

  function respond50(promptIds, opts = {}) {
    const { maxLen = 14, mood = null, recordPending = false } = opts;
    resetState();
    if (mood) setKokoro(mood);

    let sumNll = 0;
    let nllCount = 0;
    for (let t = 0; t < promptIds.length; t++) {
      const curTok = promptIds[t];
      if (t > 0 && curTok !== TOKEN_EOS) {
        const p = Math.max(1e-5, earProbs[curTok]);
        sumNll += -Math.log(p);
        nllCount += 1;
      }
      const avgNll = nllCount > 0 ? sumNll / nllCount : 0;
      currentSurprise = 1 / (1 + Math.exp(-(avgNll - 2.1) * 3.0));
      stepReservoir(curTok, currentSurprise);
      forwardEar(true);
    }

    if (recordPending && currentSurprise > 0.65) {
      pendingNovelIds = Uint8Array.from(promptIds);
      pendingNovelMood = Float32Array.from(kokoro);
    }

    const out = [];
    for (let s = 0; s < maxLen; s++) {
      forwardReadout(true);
      let nextTok = 0;
      let best = -1;
      for (let v = 0; v < VOCAB_SIZE; v++) {
        if (probs[v] > best) {
          best = probs[v];
          nextTok = v;
        }
      }
      out.push(nextTok);
      if (nextTok === TOKEN_EOS && out.length >= 2) break;
      stepReservoir(nextTok, currentSurprise * 0.8);
    }

    return {
      outIds: Uint8Array.from(out),
      surprise: Number(currentSurprise.toFixed(2)),
    };
  }

  function talkText(yamatoText, opts = {}) {
    const inIds = encode50(yamatoText, { appendEos: true });
    const { outIds, surprise } = interact50(inIds, opts);
    return { reply: decode50(outIds), surprise };
  }

  function probeText(yamatoText, opts = {}) {
    const inIds = encode50(yamatoText, { appendEos: true });
    const { outIds, surprise } = respond50(inIds, { ...opts, recordPending: false });
    return { reply: decode50(outIds), surprise };
  }

  function getFastNorm() {
    let s = 0;
    for (let i = 0; i < W_fast.length; i++) s += W_fast[i] * W_fast[i];
    return Number(Math.sqrt(s).toFixed(3));
  }

  return {
    name,
    resetState,
    setKokoro,
    getHippoSize: () => hippocampus.length,
    getFastNorm,
    trainCortexPair50,
    sleepConsolidate,
    talkText,
    probeText,
  };
}
