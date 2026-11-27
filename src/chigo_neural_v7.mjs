// chigo_neural_v7.mjs — 正真正銘・純ニューラルちご (v7)
// 特徴:
//   - 単語辞書・Map・DP分節・文字列処理はコア内に一切ゼロ！
//   - 入力ニューロン: 55個 = 50音ID (0..49) + こころ (5軸: さむ・はら・こわ・きに・やす)
//   - 隠れ層: 192個のマルチ時間スケール・リーキー再帰リザバー (Float32Array)
//     * 高速ニューロン (alpha=0.85): 1音ごとの音韻連鎖を刻む
//     * 低速ニューロン (alpha=0.22): 話題と「こころ」の文脈を保持する
//   - 出力ニューロン: 50個 (0..49 の50音そのものを1音ずつ直接発火)
//   - 学習: 予測誤差 (target - softmax) によるオンラインシナプス更新（純Float32Array）

import { VOCAB_SIZE, TOKEN_EOS, TOKEN_SEP, encode50, decode50 } from './codec50.mjs';
import { createRng } from './jomon_world.mjs';

const NUM_EMO = 5;
const IN_DIM = VOCAB_SIZE + NUM_EMO; // 55

export function createPureNeuralChigoV7(opts = {}) {
  const { name = 'ちごv7(純ニューラル)', hiddenSize = 192, seed = 77 } = opts;
  const rng = createRng(seed);
  const H = hiddenSize;
  const FEAT_DIM = H + IN_DIM + 1; // [h_t, x_t, bias]

  // 1) 入力結合 W_in (H x IN_DIM)
  const W_in = new Float32Array(H * IN_DIM);
  for (let i = 0; i < W_in.length; i++) {
    W_in[i] = (rng() * 2 - 1) * 0.55;
  }

  // 2) スパース再帰結合 W_rec (H x H, 密度15%, スペクトル半径を安定領域に調整)
  const W_rec = new Float32Array(H * H);
  const recIndices = []; // 高速化のための疎インデックス
  const recWeights = [];
  for (let i = 0; i < H; i++) {
    const idxs = [];
    const wts = [];
    for (let j = 0; j < H; j++) {
      if (rng() < 0.14) {
        const w = (rng() * 2 - 1) * 0.22;
        W_rec[i * H + j] = w;
        idxs.push(j);
        wts.push(w);
      }
    }
    recIndices.push(Int32Array.from(idxs));
    recWeights.push(Float32Array.from(wts));
  }

  // 3) マルチ時間スケール漏れ率 alpha[i]（半分は高速音韻用、半分は低速文脈用）
  const alpha = new Float32Array(H);
  for (let i = 0; i < H; i++) {
    alpha[i] = i < H / 2 ? (0.70 + rng() * 0.25) : (0.15 + rng() * 0.18);
  }

  // 4) 出力読み出し結合 W_out (50 x FEAT_DIM) + モメンタム
  const W_out = new Float32Array(VOCAB_SIZE * FEAT_DIM);
  const V_out = new Float32Array(VOCAB_SIZE * FEAT_DIM);

  // 神経状態ベクトル
  const h = new Float32Array(H);
  const hNext = new Float32Array(H);
  const feat = new Float32Array(FEAT_DIM);
  const logits = new Float32Array(VOCAB_SIZE);
  const probs = new Float32Array(VOCAB_SIZE);
  const kokoro = new Float32Array([0.2, 0.2, 0.1, 0.2, 0.5]);

  let totalTokensHeard = 0;

  function resetState() {
    h.fill(0);
  }

  function setKokoro(arr) {
    for (let i = 0; i < NUM_EMO; i++) {
      kokoro[i] = Math.max(0, Math.min(1, arr[i] ?? 0));
    }
  }

  /**
   * 1音 (tokenId: 0..49) を入力ニューロンに入れ、リザバー状態を1ステップ進める。
   */
  function stepReservoir(tokenId) {
    // 入力 x_t の非ゼロ成分: tokenId (1.0) と kokoro[0..4]
    for (let i = 0; i < H; i++) {
      const inRow = i * IN_DIM;
      let sum = W_in[inRow + tokenId];
      for (let e = 0; e < NUM_EMO; e++) {
        sum += W_in[inRow + VOCAB_SIZE + e] * kokoro[e];
      }
      // スパース再帰入力
      const idxs = recIndices[i];
      const wts = recWeights[i];
      for (let k = 0; k < idxs.length; k++) {
        sum += h[idxs[k]] * wts[k];
      }
      const a = alpha[i];
      hNext[i] = (1 - a) * h[i] + a * Math.tanh(sum);
    }
    h.set(hNext);

    // 読み出し特徴ベクトル feat = [h, onehot(tokenId), kokoro, 1.0]
    feat.fill(0);
    feat.set(h, 0);
    feat[H + tokenId] = 1.0;
    for (let e = 0; e < NUM_EMO; e++) {
      feat[H + VOCAB_SIZE + e] = kokoro[e];
    }
    feat[FEAT_DIM - 1] = 1.0;
  }

  function forwardReadout() {
    let maxL = -Infinity;
    for (let v = 0; v < VOCAB_SIZE; v++) {
      const row = v * FEAT_DIM;
      let s = 0;
      for (let k = 0; k < FEAT_DIM; k++) {
        const fk = feat[k];
        if (fk !== 0) s += W_out[row + k] * fk;
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
    for (let v = 0; v < VOCAB_SIZE; v++) {
      probs[v] /= sumExp;
    }
  }

  /**
   * 連続50音ID列を1音ずつ聴取し、次の1音の予測誤差だけで W_out をオンライン更新する。
   * (完全にトークンID列のみ。単語境界も辞書も一切与えない！)
   */
  function trainSequence50(ids, lr = 0.035) {
    if (!ids || ids.length < 2) return 0;
    let loss = 0;
    let count = 0;
    const momentum = 0.6;
    const weightDecay = 0.0002;

    for (let t = 0; t + 1 < ids.length; t++) {
      const curTok = ids[t];
      const nextTok = ids[t + 1];
      stepReservoir(curTok);
      forwardReadout();

      loss += -Math.log(Math.max(1e-7, probs[nextTok]));
      count += 1;
      totalTokensHeard += 1;

      // 出力層シナプスのオンライン更新: grad = (probs[v] - onehot(nextTok)) * feat[k]
      for (let v = 0; v < VOCAB_SIZE; v++) {
        const err = probs[v] - (v === nextTok ? 1.0 : 0.0);
        if (Math.abs(err) < 1e-4) continue;
        const row = v * FEAT_DIM;
        for (let k = 0; k < FEAT_DIM; k++) {
          const fk = feat[k];
          if (fk === 0) continue;
          const idx = row + k;
          const g = err * fk + weightDecay * W_out[idx];
          V_out[idx] = momentum * V_out[idx] - lr * g;
          W_out[idx] += V_out[idx];
        }
      }
    }
    return count > 0 ? loss / count : 0;
  }

  /**
   * 50音プロンプトを1音ずつ耳に入れ、その先の50音を出力ニューロンから1音ずつ自己回帰発火する！
   */
  function respond50(promptIds, opts = {}) {
    const { maxLen = 14, temperature = 0.05, mood = null } = opts;
    resetState();
    if (mood) setKokoro(mood);

    // 1) 入力50音を1音ずつリザバーへ流し込む
    for (let i = 0; i < promptIds.length; i++) {
      stepReservoir(promptIds[i]);
    }

    // 2) 出力ニューロン (0..49) を1音ずつ発火させて喋る
    const out = [];
    for (let s = 0; s < maxLen; s++) {
      forwardReadout();
      let nextTok = 0;
      if (temperature <= 0.08) {
        // 決定論的（最大発火ニューロン）
        let best = -1;
        for (let v = 0; v < VOCAB_SIZE; v++) {
          if (probs[v] > best) {
            best = probs[v];
            nextTok = v;
          }
        }
      } else {
        let sum = 0;
        const p = new Float32Array(VOCAB_SIZE);
        for (let v = 0; v < VOCAB_SIZE; v++) {
          p[v] = Math.pow(probs[v], 1 / temperature);
          sum += p[v];
        }
        let r = rng() * sum;
        nextTok = VOCAB_SIZE - 1;
        for (let v = 0; v < VOCAB_SIZE; v++) {
          r -= p[v];
          if (r <= 0) {
            nextTok = v;
            break;
          }
        }
      }

      out.push(nextTok);
      if (nextTok === TOKEN_EOS && out.length >= 2) break;
      stepReservoir(nextTok);
    }

    return Uint8Array.from(out);
  }

  function respondText(yamatoText, opts = {}) {
    // 発話の境界を示す「。」を末尾に付けて入力（親が喋り終えた合図）
    const inIds = encode50(yamatoText, { appendEos: true });
    const outIds = respond50(inIds, opts);
    return decode50(outIds);
  }

  return {
    name,
    resetState,
    setKokoro,
    getTotalTokensHeard: () => totalTokensHeard,
    getParamCount: () => W_in.length + W_rec.length + W_out.length,
    trainSequence50,
    respond50,
    respondText,
  };
}
