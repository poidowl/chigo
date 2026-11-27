// chigo_v12.mjs — 純ニューラルちご v12「世界モデル＋内言思考ループ（計画するGRU脳）」
// 100%純ニューラル・ゼロ依存・厳密50音I/O (0..49)
// 方策の正解ラベル（教師あり行動指示）はゼロ！
//
// v11からの大進化（AGIへの第2の壁「1発反射 → 考えてから行動する世界モデル」を突破）:
//   1. 学習可能GRU脳が「50音の世界物理」(<じょうたい>・<こうどう> => <つぎのじょうたい>) と
//      「身体価値」(<じょうたい> => V ∈ [-1, +1]) だけをエンドツーエンドBPTTで学習。
//   2. 行動を選ぶとき、1発反射で答えるのではなく、自分の出力（想像した次の50音状態）を
//      自分の耳へ戻す「内言（Inner Speech）」ロールアウトを depth 手先まで回して
//      最も未来の価値が高い行動を自力で発見・選択する！
//   3. これにより、1手目だけ得をして2手目で痛い目を見る「遅延報酬の罠」を
//      方策教師ゼロのまま頭の中のシミュレーションだけで回避できる。

import { VOCAB_SIZE, TOKEN_EOS, encode50, decode50 } from './codec50.mjs';
import { createRng } from './jomon_world.mjs';

function sigmoid(x) {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

export function createWorldModelChigoV12(opts = {}) {
  const {
    name = 'ちごv12(世界モデル+内言思考)',
    embDim = 24,
    hiddenSize = 64,
    seed = 122,
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
    W_out: initMat(VOCAB_SIZE, H, sOut),
    b_out: new Float32Array(VOCAB_SIZE),
    // 身体価値ヘッド: 50音状態の潜在ベクトル h から快/不快 [-1, +1] を推論
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

  const logits = new Float32Array(VOCAB_SIZE);
  const probs = new Float32Array(VOCAB_SIZE);

  function zeroGrads() {
    for (const k of paramKeys) grads[k].fill(0);
  }

  function softmaxOut(hVec) {
    const { W_out, b_out } = params;
    let maxL = -Infinity;
    for (let v = 0; v < VOCAB_SIZE; v++) {
      const row = v * H;
      let s = b_out[v];
      for (let j = 0; j < H; j++) s += W_out[row + j] * hVec[j];
      logits[v] = s;
      if (s > maxL) maxL = s;
    }
    let sumExp = 0;
    for (let v = 0; v < VOCAB_SIZE; v++) {
      const e = Math.exp(logits[v] - maxL);
      probs[v] = e;
      sumExp += e;
    }
    const inv = 1 / sumExp;
    for (let v = 0; v < VOCAB_SIZE; v++) probs[v] *= inv;
  }

  function forwardValue(hVec) {
    const { W_val, b_val } = params;
    let s = b_val[0];
    for (let j = 0; j < H; j++) s += W_val[j] * hVec[j];
    return Math.tanh(s);
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

  /**
   * 世界遷移 (<state>・<action> => <nextState>) の勾配を蓄積
   */
  function accumulateTransition(inIds, outIds, weight = 1.0) {
    const { W_out } = params;
    const states = [];
    let hCur = new Float32Array(H);

    for (let t = 0; t < inIds.length; t++) {
      const st = gruStepForward(inIds[t], hCur);
      states.push(st);
      hCur = st.hNext;
    }

    const outPreds = [];
    let loss = 0;
    for (let t = 0; t < outIds.length; t++) {
      softmaxOut(hCur);
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

    runBpttBack(states, dhAt);
    return outIds.length > 0 ? loss / outIds.length : 0;
  }

  /**
   * 身体価値 (<state> => targetVal ∈ [-1, +1]) の勾配を蓄積
   */
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

  function applyAdam(lr = 0.012, wd = 0.00015, clipNorm = 5.0) {
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
   * 世界物理（遷移）＋身体価値を同時学習（方策の正解は一切教えない！）
   */
  function trainWorldEpoch(transitions, stateValues, { lr = 0.012, batchSize = 8 } = {}) {
    const order = transitions.map((_, i) => i);
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = order[i];
      order[i] = order[j];
      order[j] = tmp;
    }

    let transLoss = 0;
    let valLoss = 0;
    zeroGrads();
    let count = 0;
    for (let idx = 0; idx < order.length; idx++) {
      const tr = transitions[order[idx]];
      const inIds = encode50(`${tr.state}・${tr.action}`, { appendEos: true });
      const outIds = encode50(tr.nextState, { appendEos: true });
      transLoss += accumulateTransition(inIds, outIds, 1 / batchSize);
      count += 1;
      if (count >= batchSize || idx === order.length - 1) {
        // 同じバッチに身体価値の勾配も混ぜる
        for (const sv of stateValues) {
          const sIds = encode50(sv.state, { appendEos: true });
          valLoss += accumulateValue(sIds, sv.value, 0.5 / stateValues.length);
        }
        applyAdam(lr);
        count = 0;
      }
    }
    return {
      transLoss: transitions.length > 0 ? transLoss / transitions.length : 0,
      valLoss: stateValues.length > 0 ? valLoss / stateValues.length : 0,
    };
  }

  /**
   * 想像（世界モデル1手先予測）: 50音状態 + 50音行動 => 想像される次の50音状態
   */
  function imagineNextState(stateText, actionText, maxLen = 12) {
    const cleanState = String(stateText).replace(/。+$/g, '');
    const inIds = encode50(`${cleanState}・${actionText}`, { appendEos: true });
    let hCur = new Float32Array(H);
    for (let t = 0; t < inIds.length; t++) hCur = gruStepForward(inIds[t], hCur).hNext;

    const out = [];
    for (let s = 0; s < maxLen; s++) {
      softmaxOut(hCur);
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

  /**
   * 身体価値の推論: 50音状態 => V(s) ∈ [-1, +1]
   */
  function estimateValue(stateText) {
    const cleanState = String(stateText).replace(/。+$/g, '');
    const sIds = encode50(cleanState, { appendEos: true });
    let hCur = new Float32Array(H);
    for (let t = 0; t < sIds.length; t++) hCur = gruStepForward(sIds[t], hCur).hNext;
    return Number(forwardValue(hCur).toFixed(3));
  }

  /**
   * 内言思考ループ (Inner-Speech Mental Rollout):
   * 自分の世界モデルを使って depth 手先まで50音シミュレーションし、
   * 最も未来の価値が高い行動を自力で選ぶ。
   */
  function thinkAndPlan(stateText, candidateActions, { depth = 2, gamma = 0.92 } = {}) {
    function search(curState, remDepth) {
      let bestScore = -Infinity;
      let bestPlan = null;

      for (const act of candidateActions) {
        const nextState = imagineNextState(curState, act);
        const vImmediate = estimateValue(nextState);

        let totalScore = vImmediate;
        let trace = [{ state: curState, action: act, imaginedNext: nextState, value: vImmediate }];

        // 終端状態（非常に良い or 非常に悪い）でなければさらに頭の中で先読みする
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

  function countParams() {
    let n = 0;
    for (const k of paramKeys) n += params[k].length;
    return n;
  }

  return {
    name,
    embDim: D,
    hiddenSize: H,
    countParams,
    trainWorldEpoch,
    imagineNextState,
    estimateValue,
    thinkAndPlan,
  };
}
