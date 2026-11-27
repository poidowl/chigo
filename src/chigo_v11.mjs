// chigo_v11.mjs — 純ニューラルちご v11「固定リザバー卒業・自己概念形成GRU脳 (End-to-End BPTT)」
// 100%純ニューラル・ゼロ依存・厳密50音I/O (0..49)
// 単語辞書・カテゴリラベル・Map・if文定型返答は一切ゼロ！
//
// v10からの大進化（AGIへの第1の壁「固定リザバー」を突破）:
//   1. 学習可能な50音埋め込み (E: 50 × D) ＋ ゲート付き再帰脳 (GRU: H次元) を
//      時間展開誤差逆伝播 (BPTT) + Adam でエンドツーエンド学習。
//   2. 内部表現そのものが自己組織化するため、文字が1文字も被らない語
//      （例: 「しか」と「いも」、「くま」と「へび」）が同じ潜在概念クラスターへ自然に集まり、
//      訓練で一度も見たことのない組み合わせ（例: 「いも・ない」「へび・ない」）を
//      ゼロショットで構成的汎化（Compositional Generalization）できる！
//   3. 1回教えただけの未知語（例: 「うさぎ・ある => とる。」）を海馬リプレイ付きBPTTで
//      潜在概念空間へ着地させ、未教示の文脈（「うさぎ・ない => さがす。」）へ即座に転用できる！

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

/**
 * 初期知識ゼロの「自己概念形成 GRU ちご v11」を生成する。
 */
export function createConceptChigoV11(opts = {}) {
  const {
    name = 'ちごv11(自己概念形成GRU)',
    embDim = 24,
    hiddenSize = 64,
    seed = 111,
  } = opts;

  const rng = createRng(seed);
  const D = embDim;
  const H = hiddenSize;

  function initMat(rows, cols, scale) {
    const arr = new Float32Array(rows * cols);
    for (let i = 0; i < arr.length; i++) {
      arr[i] = (rng() * 2 - 1) * scale;
    }
    return arr;
  }

  const sIn = Math.sqrt(1 / D) * 0.8;
  const sRec = Math.sqrt(1 / H) * 0.7;
  const sOut = Math.sqrt(1 / H) * 0.6;

  // 全学習可能パラメータ (Float32Array のみ)
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
    W_ear: initMat(VOCAB_SIZE, H, sOut),
    b_ear: new Float32Array(VOCAB_SIZE),
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

  // 海馬エピソード記憶（生の50音ID列のみ）
  const hippocampus = [];
  const MAX_HIPPO = 128;

  // 作業バッファ
  const logits = new Float32Array(VOCAB_SIZE);
  const probs = new Float32Array(VOCAB_SIZE);
  const earProbs = new Float32Array(VOCAB_SIZE);

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

  function recordHippo(inIds, outIds, isNew = false) {
    const key = Array.from(inIds).join(',');
    const idx = hippocampus.findIndex((e) => Array.from(e.inIds).join(',') === key);
    const entry = { inIds: Uint8Array.from(inIds), outIds: Uint8Array.from(outIds), isNew };
    if (idx >= 0) hippocampus[idx] = entry;
    else {
      hippocampus.push(entry);
      if (hippocampus.length > MAX_HIPPO) hippocampus.shift();
    }
  }

  /**
   * 1エピソード (inIds -> outIds) についてエンドツーエンド BPTT で勾配を蓄積する。
   */
  function accumulateBpttPair(inIds, outIds, { trainEar = true, weight = 1.0 } = {}) {
    const { E, W_zx, W_zh, W_rx, W_rh, W_nx, W_nh, W_out, b_out, W_ear, b_ear } = params;
    const states = [];
    let hCur = new Float32Array(H);

    // 1) 聴取フェーズ (inIds)
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

    // 2) 発話フェーズ (outIds: teacher forcing)
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

    // 各時刻の dh を初期化
    const T = states.length;
    const dhAt = Array.from({ length: T }, () => new Float32Array(H));

    // 発話読み出し (W_out, b_out) の勾配
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

    // 聴覚予測 (W_ear, b_ear) の勾配（驚き検出用・本体概念形成を邪魔しないよう軽めに逆伝播）
    const earScale = 0.25 * weight;
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
          dh[j] += W_ear[row + j] * err * 0.2;
        }
      }
    }

    // 3) 時間逆伝播 (BPTT: t = T-1 .. 0)
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

      // drh = W_nh^T @ da_n
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

      // dhCarry += W_zh^T @ da_z + W_rh^T @ da_r
      // dE[tokId] += W_zx^T @ da_z + W_rx^T @ da_r + W_nx^T @ da_n
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

    return outIds.length > 0 ? loss / outIds.length : 0;
  }

  function applyAdam(lr = 0.012, wd = 0.00015, clipNorm = 5.0, onlyKeys = null) {
    adamStep += 1;
    const b1 = 0.9;
    const b2 = 0.999;
    const eps = 1e-8;
    const bc1 = 1 - Math.pow(b1, adamStep);
    const bc2 = 1 - Math.pow(b2, adamStep);

    const activeKeys = onlyKeys || paramKeys;

    // Global L2 norm gradient clipping
    let normSq = 0;
    for (const k of activeKeys) {
      const g = grads[k];
      for (let i = 0; i < g.length; i++) normSq += g[i] * g[i];
    }
    const norm = Math.sqrt(normSq);
    const scale = norm > clipNorm ? clipNorm / norm : 1.0;

    for (const k of activeKeys) {
      const p = params[k];
      const g = grads[k];
      const m = adamM[k];
      const v = adamV[k];
      const isBias = k.startsWith('b_');
      for (let i = 0; i < p.length; i++) {
        const gi = g[i] * scale + (isBias ? 0 : wd * p[i]);
        m[i] = b1 * m[i] + (1 - b1) * gi;
        v[i] = b2 * v[i] + (1 - b2) * gi * gi;
        const mHat = m[i] / bc1;
        const vHat = v[i] / bc2;
        p[i] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
      }
    }
    zeroGrads();
  }

  /**
   * 幼少期バッチ学習（BPTT + Adam）
   */
  function trainEpoch(pairs, { lr = 0.012, batchSize = 8, saveToHippo = false } = {}) {
    // Fisher-Yates shuffle
    const order = pairs.map((_, i) => i);
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = order[i];
      order[i] = order[j];
      order[j] = tmp;
    }

    let totalLoss = 0;
    zeroGrads();
    let countInBatch = 0;
    for (let idx = 0; idx < order.length; idx++) {
      const item = pairs[order[idx]];
      const inIds = item.inIds || encode50(item.in, { appendEos: true });
      const outIds = item.outIds || encode50(item.out, { appendEos: true });
      if (saveToHippo) recordHippo(inIds, outIds, false);
      totalLoss += accumulateBpttPair(inIds, outIds, { trainEar: true, weight: 1 / batchSize });
      countInBatch += 1;
      if (countInBatch >= batchSize || idx === order.length - 1) {
        applyAdam(lr);
        countInBatch = 0;
      }
    }
    return pairs.length > 0 ? totalLoss / pairs.length : 0;
  }

  /**
   * 1ショット概念誘導（未知語を1例だけ教えたとき、海馬リプレイと混ぜながら
   * 内部表現へ着地させ、未教示の文脈にも構成的汎化させる！）
   */
  function learnConceptOneShot(inText, outText, { steps = 28, lr = 0.008 } = {}) {
    const inIds = encode50(inText, { appendEos: true });
    const outIds = encode50(outText, { appendEos: true });
    recordHippo(inIds, outIds, true);

    const newItems = hippocampus.filter((e) => e.isNew);
    const oldItems = hippocampus.filter((e) => !e.isNew);

    for (let s = 0; s < steps; s++) {
      zeroGrads();
      // 新規エピソードを強めに、既存記憶をアンカーとして混ぜてBPTT
      for (const e of newItems) {
        accumulateBpttPair(e.inIds, e.outIds, { trainEar: true, weight: 0.45 });
      }
      const anchorCount = Math.min(12, oldItems.length);
      for (let a = 0; a < anchorCount; a++) {
        const pick = oldItems[Math.floor(rng() * oldItems.length)];
        accumulateBpttPair(pick.inIds, pick.outIds, { trainEar: false, weight: 1.0 / anchorCount });
      }
      applyAdam(lr, 0.0001, 4.0);
    }
  }

  /**
   * 50音列を聞いた直後の潜在概念ベクトル h を取り出す（概念クラスター可視化・検証用）
   */
  function encodeConceptVector(kanaText) {
    const ids = encode50(kanaText, { appendEos: false });
    let hCur = new Float32Array(H);
    for (let t = 0; t < ids.length; t++) {
      hCur = gruStepForward(ids[t], hCur).hNext;
    }
    return hCur;
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
    const surprise = 1 / (1 + Math.exp(-(avgNll - 2.6) * 2.5));

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
    };
  }

  function probeText(kanaText, opts = {}) {
    const inIds = encode50(kanaText, { appendEos: true });
    const { outIds, surprise } = respond50(inIds, opts);
    return { reply: decode50(outIds), surprise };
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
    getHippoSize: () => hippocampus.length,
    trainEpoch,
    learnConceptOneShot,
    encodeConceptVector,
    probeText,
  };
}

export function cosineSim(a, b) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na * nb);
  return denom > 1e-8 ? dot / denom : 0;
}
