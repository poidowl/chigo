// chigo_v14.mjs — 統合純ニューラルちご v14「大脳皮質(GRU) × 海馬(リプレイ) × 基底核(Double-DQN)」
// 100%純ニューラル・ゼロ依存・厳密50音I/O (0..49)
//
// v10〜v13で単独突破した4つの能力を、1つの統合個体へ合体:
//   [大脳皮質: 共有学習可能GRU (80ユニット, End-to-End BPTT + Adam)]
//     1. かな会話＆自律「こころ」回路 (W_emo: 5軸情動)
//     2. 自己概念形成＆構成的汎化＋1ショット未知語概念転用
//     3. 世界モデル(遷移予測)＋身体価値(W_val)＋内言思考ループ(depth=2)
//     4. 聴覚予測誤差「驚き」ニューロン (W_ear → 未知語で「なに。」)
//   [海馬: 50音エピソード記憶バッファ]
//     - オンライン未知語習得・1ショット概念誘導時の交互リプレイで破滅的忘却を防止
//   [基底核: 50音スパース Double-DQN]
//     - 凍結v0縄文環境で観測テキストのdecode・ルール・方策教師ゼロのまま報酬サバイバル

import { VOCAB_SIZE, TOKEN_SEP, TOKEN_EOS, encode50, decode50 } from './codec50.mjs';
import { createRng } from './jomon_world.mjs';
import { createSurvivalChigoV13 } from './chigo_v13.mjs';

export const EMO_NAMES = ['さむ', 'はら', 'こわ', 'きに', 'やす'];
const NUM_EMO = EMO_NAMES.length;

function sigmoid(x) {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

/**
 * 1つの統合純ニューラル個体「ちご v14」を生成する。
 */
export function createUnifiedChigoV14(opts = {}) {
  const {
    name = 'ちごv14(統合純ニューラルAGI)',
    embDim = 24,
    hiddenSize = 80,
    seed = 1400,
  } = opts;

  const rng = createRng(seed);
  const D = embDim;
  const H = hiddenSize;

  function initMat(rows, cols, scale) {
    const arr = new Float32Array(rows * cols);
    for (let i = 0; i < arr.length; i++) arr[i] = (rng() * 2 - 1) * scale;
    return arr;
  }

  const sIn = Math.sqrt(1 / D) * 0.8;
  const sRec = Math.sqrt(1 / H) * 0.7;
  const sOut = Math.sqrt(1 / H) * 0.6;

  // === 1. 大脳皮質 (共有 GRU + 4つの読み出しヘッド) ===
  const params = {
    E: initMat(VOCAB_SIZE, D, 0.35),
    W_zx: initMat(H, D, sIn),
    W_zh: initMat(H, H, sRec),
    b_z: new Float32Array(H),
    W_rx: initMat(H, D, sIn),
    W_rh: initMat(H, H, sRec),
    b_r: new Float32Array(H),
    W_nx: initMat(H, D, sIn),
    W_nh: initMat(H, H, sRec),
    b_n: new Float32Array(H),
    // 言語・概念・世界遷移 共通出力ヘッド
    W_out: initMat(VOCAB_SIZE, H, sOut),
    b_out: new Float32Array(VOCAB_SIZE),
    // 聴覚予測ヘッド (驚き検出)
    W_ear: initMat(VOCAB_SIZE, H, sOut),
    b_ear: new Float32Array(VOCAB_SIZE),
    // 自律こころヘッド (5軸情動)
    W_emo: initMat(NUM_EMO, H, sOut),
    b_emo: new Float32Array(NUM_EMO),
    // 身体価値ヘッド (内言ロールアウト用 V(s) ∈ [-1, +1])
    W_val: initMat(1, H, sOut),
    b_val: new Float32Array(1),
  };

  const paramKeys = Object.keys(params);
  const grads = {};
  const adamM = {};
  const adamV = {};
  for (const k of paramKeys) {
    grads[k] = new Float32Array(params[k].length);
    adamM[k] = new Float32Array(params[k].length);
    adamV[k] = new Float32Array(params[k].length);
  }
  let adamStep = 0;

  // === 2. 海馬 (エピソード記憶バッファ) ===
  const hippocampus = [];
  const MAX_HIPPO = 200;

  // === 3. 基底核 (無教師・報酬サバイバル運動系) ===
  const basalGanglia = createSurvivalChigoV13({
    name: `${name}:基底核`,
    seed: 1337,
  });

  // 状態バッファ
  const logits = new Float32Array(VOCAB_SIZE);
  const probs = new Float32Array(VOCAB_SIZE);
  const earProbs = new Float32Array(VOCAB_SIZE);
  const kokoro = new Float32Array([0.2, 0.2, 0.1, 0.3, 0.6]);

  function zeroGrads() {
    for (const k of paramKeys) grads[k].fill(0);
  }

  function softmaxInto(W, b, hVec, outProbs) {
    let maxL = -Infinity;
    for (let v = 0; v < VOCAB_SIZE; v++) {
      const row = v * H;
      let s = b[v];
      for (let j = 0; j < H; j++) s += W[row + j] * hVec[j];
      logits[v] = s;
      if (s > maxL) maxL = s;
    }
    let sumExp = 0;
    for (let v = 0; v < VOCAB_SIZE; v++) {
      const e = Math.exp(logits[v] - maxL);
      outProbs[v] = e;
      sumExp += e;
    }
    const inv = 1 / sumExp;
    for (let v = 0; v < VOCAB_SIZE; v++) outProbs[v] *= inv;
  }

  function forwardValue(hVec) {
    const { W_val, b_val } = params;
    let s = b_val[0];
    for (let j = 0; j < H; j++) s += W_val[j] * hVec[j];
    return Math.tanh(s);
  }

  function forwardEmo(hVec) {
    const { W_emo, b_emo } = params;
    const out = new Float32Array(NUM_EMO);
    for (let e = 0; e < NUM_EMO; e++) {
      const row = e * H;
      let s = b_emo[e];
      for (let j = 0; j < H; j++) s += W_emo[row + j] * hVec[j];
      out[e] = sigmoid(s);
    }
    return out;
  }

  function gruStepForward(tokId, hPrev) {
    const { E, W_zx, W_zh, b_z, W_rx, W_rh, b_r, W_nx, W_nh, b_n } = params;
    const xOff = tokId * D;
    const z = new Float32Array(H);
    const r = new Float32Array(H);
    const rh = new Float32Array(H);
    const n = new Float32Array(H);
    const hNext = new Float32Array(H);

    for (let i = 0; i < H; i++) {
      const rowX = i * D;
      const rowH = i * H;
      let sz = b_z[i];
      let sr = b_r[i];
      for (let j = 0; j < D; j++) {
        const xv = E[xOff + j];
        sz += W_zx[rowX + j] * xv;
        sr += W_rx[rowX + j] * xv;
      }
      for (let j = 0; j < H; j++) {
        const hv = hPrev[j];
        sz += W_zh[rowH + j] * hv;
        sr += W_rh[rowH + j] * hv;
      }
      z[i] = sigmoid(sz);
      r[i] = sigmoid(sr);
      rh[i] = r[i] * hPrev[i];
    }

    for (let i = 0; i < H; i++) {
      const rowX = i * D;
      const rowH = i * H;
      let sn = b_n[i];
      for (let j = 0; j < D; j++) sn += W_nx[rowX + j] * E[xOff + j];
      for (let j = 0; j < H; j++) sn += W_nh[rowH + j] * rh[j];
      n[i] = Math.tanh(sn);
      hNext[i] = (1 - z[i]) * hPrev[i] + z[i] * n[i];
    }

    return { tokId, hPrev, z, r, rh, n, hNext };
  }

  function runBpttBack(states, dhAt) {
    const { E, W_zx, W_zh, W_rx, W_rh, W_nx, W_nh } = params;
    const T = states.length;
    const dhCarry = new Float32Array(H);
    const da_z = new Float32Array(H);
    const da_r = new Float32Array(H);
    const da_n = new Float32Array(H);
    const drh = new Float32Array(H);

    for (let t = T - 1; t >= 0; t--) {
      const st = states[t];
      const { tokId, hPrev, z, r, rh, n } = st;
      const dh = dhAt[t];
      for (let i = 0; i < H; i++) dh[i] += dhCarry[i];
      dhCarry.fill(0);

      for (let i = 0; i < H; i++) {
        const dhi = dh[i];
        const dn = dhi * z[i];
        da_n[i] = dn * (1 - n[i] * n[i]);
        const dz = dhi * (n[i] - hPrev[i]);
        da_z[i] = dz * z[i] * (1 - z[i]);
        dhCarry[i] += dhi * (1 - z[i]);
      }

      drh.fill(0);
      for (let i = 0; i < H; i++) {
        const dan = da_n[i];
        if (dan === 0) continue;
        const rowH = i * H;
        for (let j = 0; j < H; j++) drh[j] += W_nh[rowH + j] * dan;
      }

      for (let i = 0; i < H; i++) {
        const dr = drh[i] * hPrev[i];
        da_r[i] = dr * r[i] * (1 - r[i]);
        dhCarry[i] += drh[i] * r[i];
      }

      const xOff = tokId * D;
      for (let i = 0; i < H; i++) {
        const daz = da_z[i];
        const dar = da_r[i];
        const dan = da_n[i];
        grads.b_z[i] += daz;
        grads.b_r[i] += dar;
        grads.b_n[i] += dan;

        const rowX = i * D;
        const rowH = i * H;
        for (let j = 0; j < D; j++) {
          const xv = E[xOff + j];
          grads.W_zx[rowX + j] += daz * xv;
          grads.W_rx[rowX + j] += dar * xv;
          grads.W_nx[rowX + j] += dan * xv;
          grads.E[xOff + j] += W_zx[rowX + j] * daz + W_rx[rowX + j] * dar + W_nx[rowX + j] * dan;
        }
        for (let j = 0; j < H; j++) {
          const hv = hPrev[j];
          grads.W_zh[rowH + j] += daz * hv;
          grads.W_rh[rowH + j] += dar * hv;
          grads.W_nh[rowH + j] += dan * rh[j];
          dhCarry[j] += W_zh[rowH + j] * daz + W_rh[rowH + j] * dar;
        }
      }
    }
  }

  function recordHippo(inIds, outIds, mood = null, isNew = false) {
    const key = Array.from(inIds).join(',');
    const idx = hippocampus.findIndex((e) => Array.from(e.inIds).join(',') === key);
    const entry = {
      inIds: Uint8Array.from(inIds),
      outIds: Uint8Array.from(outIds),
      mood: mood ? Float32Array.from(mood) : null,
      isNew,
    };
    if (idx >= 0) hippocampus[idx] = entry;
    else {
      hippocampus.push(entry);
      if (hippocampus.length > MAX_HIPPO) hippocampus.shift();
    }
  }

  function accumulateSeqPair(inIds, outIds, { mood = null, trainEar = true, weight = 1.0 } = {}) {
    const { W_out, b_out, W_ear, b_ear, W_emo } = params;
    const states = [];
    let hCur = new Float32Array(H);

    const earPreds = [];
    for (let t = 0; t < inIds.length; t++) {
      const st = gruStepForward(inIds[t], hCur);
      states.push(st);
      hCur = st.hNext;
      if (trainEar && t + 1 < inIds.length) {
        softmaxInto(W_ear, b_ear, hCur, earProbs);
        earPreds.push({ stateIdx: states.length - 1, targetTok: inIds[t + 1], probs: Float32Array.from(earProbs) });
      }
    }

    const inputEndIdx = states.length - 1;
    const outPreds = [];
    let loss = 0;
    for (let t = 0; t < outIds.length; t++) {
      softmaxInto(W_out, b_out, hCur, probs);
      const targetTok = outIds[t];
      loss += -Math.log(Math.max(1e-7, probs[targetTok]));
      outPreds.push({ stateIdx: states.length - 1, targetTok, probs: Float32Array.from(probs) });
      if (t + 1 < outIds.length) {
        const st = gruStepForward(targetTok, hCur);
        states.push(st);
        hCur = st.hNext;
      }
    }

    const dhAt = Array.from({ length: states.length }, () => new Float32Array(H));

    // 出力ヘッド勾配
    for (const p of outPreds) {
      const hVec = states[p.stateIdx].hNext;
      const dh = dhAt[p.stateIdx];
      for (let v = 0; v < VOCAB_SIZE; v++) {
        const err = (p.probs[v] - (v === p.targetTok ? 1.0 : 0.0)) * weight;
        if (Math.abs(err) < 1e-6) continue;
        grads.b_out[v] += err;
        const row = v * H;
        for (let j = 0; j < H; j++) {
          grads.W_out[row + j] += err * hVec[j];
          dh[j] += W_out[row + j] * err;
        }
      }
    }

    // 聴覚予測ヘッド勾配
    const earScale = 0.22 * weight;
    for (const p of earPreds) {
      const hVec = states[p.stateIdx].hNext;
      const dh = dhAt[p.stateIdx];
      for (let v = 0; v < VOCAB_SIZE; v++) {
        const err = (p.probs[v] - (v === p.targetTok ? 1.0 : 0.0)) * earScale;
        if (Math.abs(err) < 1e-6) continue;
        grads.b_ear[v] += err;
        const row = v * H;
        for (let j = 0; j < H; j++) {
          grads.W_ear[row + j] += err * hVec[j];
          dh[j] += W_ear[row + j] * err * 0.15;
        }
      }
    }

    // 自律こころヘッド勾配
    if (mood) {
      const hIn = states[inputEndIdx].hNext;
      const emoPred = forwardEmo(hIn);
      const dh = dhAt[inputEndIdx];
      const emoScale = 0.35 * weight;
      for (let e = 0; e < NUM_EMO; e++) {
        const err = (emoPred[e] - mood[e]) * emoScale;
        grads.b_emo[e] += err;
        const row = e * H;
        for (let j = 0; j < H; j++) {
          grads.W_emo[row + j] += err * hIn[j];
          dh[j] += W_emo[row + j] * err * 0.15;
        }
      }
    }

    runBpttBack(states, dhAt);
    return outIds.length > 0 ? loss / outIds.length : 0;
  }

  function accumulateValue(stateIds, targetVal, weight = 1.0) {
    const { W_val } = params;
    const states = [];
    let hCur = new Float32Array(H);
    for (let t = 0; t < stateIds.length; t++) {
      const st = gruStepForward(stateIds[t], hCur);
      states.push(st);
      hCur = st.hNext;
    }
    const pred = forwardValue(hCur);
    const diff = pred - targetVal;
    const loss = 0.5 * diff * diff;
    const dRaw = diff * (1 - pred * pred) * weight;

    const dhAt = Array.from({ length: states.length }, () => new Float32Array(H));
    const lastDh = dhAt[states.length - 1];
    grads.b_val[0] += dRaw;
    for (let j = 0; j < H; j++) {
      grads.W_val[j] += dRaw * hCur[j];
      lastDh[j] += W_val[j] * dRaw;
    }
    runBpttBack(states, dhAt);
    return loss;
  }

  function applyAdam(lr = 0.012, wd = 0.00012, clipNorm = 5.0) {
    adamStep += 1;
    const b1 = 0.9;
    const b2 = 0.999;
    const eps = 1e-8;
    const bc1 = 1 - Math.pow(b1, adamStep);
    const bc2 = 1 - Math.pow(b2, adamStep);

    let normSq = 0;
    for (const k of paramKeys) {
      const g = grads[k];
      for (let i = 0; i < g.length; i++) normSq += g[i] * g[i];
    }
    const norm = Math.sqrt(normSq);
    const scale = norm > clipNorm ? clipNorm / norm : 1.0;

    for (const k of paramKeys) {
      const p = params[k];
      const g = grads[k];
      const m = adamM[k];
      const v = adamV[k];
      const isBias = k.startsWith('b_');
      for (let i = 0; i < p.length; i++) {
        const gi = g[i] * scale + (isBias ? 0 : wd * p[i]);
        m[i] = b1 * m[i] + (1 - b1) * gi;
        v[i] = b2 * v[i] + (1 - b2) * gi * gi;
        p[i] -= (lr * (m[i] / bc1)) / (Math.sqrt(v[i] / bc2) + eps);
      }
    }
    zeroGrads();
  }

  /**
   * 統合大脳皮質の1エポック学習（会話＋概念＋世界物理＋身体価値を1つのGRUで同時習得！）
   */
  function trainUnifiedCortexEpoch(seqPairs, stateValues = [], { lr = 0.012, batchSize = 10, saveToHippo = false } = {}) {
    const order = seqPairs.map((_, i) => i);
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = order[i];
      order[i] = order[j];
      order[j] = tmp;
    }

    let totalLoss = 0;
    zeroGrads();
    let count = 0;
    for (let idx = 0; idx < order.length; idx++) {
      const item = seqPairs[order[idx]];
      const inIds = item.inIds || encode50(item.in, { appendEos: true });
      const outIds = item.outIds || encode50(item.out, { appendEos: true });
      if (saveToHippo) recordHippo(inIds, outIds, item.mood || null, false);
      totalLoss += accumulateSeqPair(inIds, outIds, {
        mood: item.mood || null,
        trainEar: true,
        weight: 1 / batchSize,
      });
      count += 1;
      if (count >= batchSize || idx === order.length - 1) {
        if (stateValues.length > 0) {
          for (const sv of stateValues) {
            accumulateValue(encode50(sv.state, { appendEos: true }), sv.value, 0.5 / stateValues.length);
          }
        }
        applyAdam(lr);
        count = 0;
      }
    }
    return seqPairs.length > 0 ? totalLoss / seqPairs.length : 0;
  }

  /**
   * 1ショット未知語概念誘導（海馬リプレイと混ぜてBPTT → 既存記憶を壊さず未経験文脈へ転用！）
   */
  function learnConceptOneShot(inText, outText, { steps = 28, lr = 0.0045 } = {}) {
    const inIds = encode50(inText, { appendEos: true });
    const outIds = encode50(outText, { appendEos: true });
    recordHippo(inIds, outIds, null, true);

    const newItems = hippocampus.filter((e) => e.isNew);
    const oldItems = hippocampus.filter((e) => !e.isNew);

    for (let s = 0; s < steps; s++) {
      zeroGrads();
      for (const e of newItems) {
        accumulateSeqPair(e.inIds, e.outIds, { trainEar: true, weight: 0.35 });
      }
      const anchorCount = Math.min(28, oldItems.length);
      for (let a = 0; a < anchorCount; a++) {
        const pick = oldItems[Math.floor(rng() * oldItems.length)];
        accumulateSeqPair(pick.inIds, pick.outIds, { mood: pick.mood, trainEar: false, weight: 1.0 / anchorCount });
      }
      applyAdam(lr, 0.0001, 4.0);
    }
  }

  function respond50(promptIds, { maxLen = 14 } = {}) {
    const { W_out, b_out, W_ear, b_ear } = params;
    let hCur = new Float32Array(H);
    let sumNll = 0;
    let nllCount = 0;

    for (let t = 0; t < promptIds.length; t++) {
      const curTok = promptIds[t];
      if (t > 0 && curTok !== TOKEN_EOS) {
        sumNll += -Math.log(Math.max(1e-5, earProbs[curTok]));
        nllCount += 1;
      }
      hCur = gruStepForward(curTok, hCur).hNext;
      softmaxInto(W_ear, b_ear, hCur, earProbs);
    }

    const avgNll = nllCount > 0 ? sumNll / nllCount : 0;
    const surprise = 1 / (1 + Math.exp(-(avgNll - 3.3) * 2.5));

    // 自律こころ更新
    const emoPred = forwardEmo(hCur);
    for (let e = 0; e < NUM_EMO; e++) {
      kokoro[e] = kokoro[e] * 0.25 + emoPred[e] * 0.75;
    }

    // 未知語への驚き発火（構成的組み合わせは通しつつ、真の未知音列に「なに。」を返す）
    if (surprise > 0.72) {
      return {
        outIds: encode50('なに。', { appendEos: true }),
        surprise: Number(surprise.toFixed(2)),
        kokoro: Array.from(kokoro, (v) => Number(v.toFixed(2))),
      };
    }

    const out = [];
    for (let s = 0; s < maxLen; s++) {
      softmaxInto(W_out, b_out, hCur, probs);
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
      hCur = gruStepForward(nextTok, hCur).hNext;
    }

    return {
      outIds: Uint8Array.from(out),
      surprise: Number(surprise.toFixed(2)),
      kokoro: Array.from(kokoro, (v) => Number(v.toFixed(2))),
    };
  }

  function probeText(kanaText, opts = {}) {
    const inIds = encode50(kanaText, { appendEos: true });
    const res = respond50(inIds, opts);
    return { reply: decode50(res.outIds), surprise: res.surprise, kokoro: res.kokoro };
  }

  function imagineNextState(stateText, actionText, maxLen = 12) {
    const cleanState = String(stateText).replace(/。+$/g, '');
    const inIds = encode50(`${cleanState}・${actionText}`, { appendEos: true });
    let hCur = new Float32Array(H);
    for (let t = 0; t < inIds.length; t++) hCur = gruStepForward(inIds[t], hCur).hNext;

    const out = [];
    for (let s = 0; s < maxLen; s++) {
      softmaxInto(params.W_out, params.b_out, hCur, probs);
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
      hCur = gruStepForward(nextTok, hCur).hNext;
    }
    return decode50(Uint8Array.from(out)).replace(/。+$/g, '');
  }

  function estimateValue(stateText) {
    const cleanState = String(stateText).replace(/。+$/g, '');
    const sIds = encode50(cleanState, { appendEos: true });
    let hCur = new Float32Array(H);
    for (let t = 0; t < sIds.length; t++) hCur = gruStepForward(sIds[t], hCur).hNext;
    return Number(forwardValue(hCur).toFixed(3));
  }

  function thinkAndPlan(stateText, candidateActions, { depth = 2, gamma = 0.92 } = {}) {
    function search(curState, remDepth) {
      let bestScore = -Infinity;
      let bestPlan = null;
      for (const act of candidateActions) {
        const nextState = imagineNextState(curState, act);
        const vImmediate = estimateValue(nextState);
        let totalScore = vImmediate;
        let trace = [{ state: curState, action: act, imaginedNext: nextState, value: vImmediate }];
        if (remDepth > 1 && Math.abs(vImmediate) < 0.85) {
          const sub = search(nextState, remDepth - 1);
          totalScore = vImmediate + gamma * sub.score;
          trace = trace.concat(sub.trace);
        }
        if (totalScore > bestScore) {
          bestScore = totalScore;
          bestPlan = { action: act, score: Number(totalScore.toFixed(3)), trace };
        }
      }
      return bestPlan;
    }
    return search(stateText, depth);
  }

  function countCortexParams() {
    let n = 0;
    for (const k of paramKeys) n += params[k].length;
    return n;
  }

  return {
    name,
    countParams: () => ({
      cortex: countCortexParams(),
      basalGanglia: basalGanglia.countParams(),
      total: countCortexParams() + basalGanglia.countParams(),
    }),
    getHippoSize: () => hippocampus.length,
    // 大脳皮質API（会話・概念・世界モデル・内言思考）
    trainUnifiedCortexEpoch,
    learnConceptOneShot,
    probeText,
    imagineNextState,
    estimateValue,
    thinkAndPlan,
    // 基底核API（縄文サバイバルRL）
    setEvalMode: basalGanglia.setEvalMode,
    reset: basalGanglia.reset,
    act: basalGanglia.act,
    learn: basalGanglia.learn,
  };
}
