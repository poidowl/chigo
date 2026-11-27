// chigo_neural_v8.mjs — 純ニューラルちご v8 (予測誤差「驚き」回路 + 3時間スケール作業記憶リザバー)
// 100%純ニューラル: 単語辞書・Map・DP分節・if文による定型返答はゼロ！
//
// v7からの進化点:
//   1. 3時間スケール・リーキーリザバー (H = 240):
//      - 高速 (alpha=0.85): 1音ごとの正確な50音発声
//      - 中速 (alpha=0.35): 単語レベルの音韻まとまり保持
//      - 低速 (alpha=0.10): 複合入力（「あめ・くま」等）の前半概念を保持する作業記憶
//   2. 予測誤差（驚き / Surprise）ニューロン:
//      入力50音を1音ずつ聞くたびに自前の読み出し確率 probs[tok] から
//      -log(probs[tok]) を計算し、入力ニューロン(56番目)へフィードバック。
//      未知の音の並びが来ると驚きニューロンが強く発火し、学習した回路で「なに。」を出力する。

import { VOCAB_SIZE, TOKEN_EOS, encode50, decode50 } from './codec50.mjs';
import { createRng } from './jomon_world.mjs';

const NUM_EMO = 5;
const IN_DIM = VOCAB_SIZE + NUM_EMO + 1; // 50音 + こころ5軸 + 驚き1軸 = 56
const IDX_SURPRISE = VOCAB_SIZE + NUM_EMO;

export function createPureNeuralChigoV8(opts = {}) {
  const { name = 'ちごv8(純ニューラル+驚き)', hiddenSize = 240, seed = 88 } = opts;
  const rng = createRng(seed);
  const H = hiddenSize;
  const FEAT_DIM = H + IN_DIM + 1; // [h_t, x_t(56), bias]

  // 入力結合 W_in (H x IN_DIM)
  const W_in = new Float32Array(H * IN_DIM);
  for (let i = 0; i < W_in.length; i++) {
    W_in[i] = (rng() * 2 - 1) * 0.5;
  }

  // スパース再帰結合 W_rec (密度12%)
  const recIndices = [];
  const recWeights = [];
  let recParamCount = 0;
  for (let i = 0; i < H; i++) {
    const idxs = [];
    const wts = [];
    for (let j = 0; j < H; j++) {
      if (rng() < 0.12) {
        idxs.push(j);
        wts.push((rng() * 2 - 1) * 0.2);
        recParamCount += 1;
      }
    }
    recIndices.push(Int32Array.from(idxs));
    recWeights.push(Float32Array.from(wts));
  }

  // 3時間スケール漏れ率 alpha[i]
  const alpha = new Float32Array(H);
  const third = Math.floor(H / 3);
  for (let i = 0; i < H; i++) {
    if (i < third) alpha[i] = 0.78 + rng() * 0.18;          // 高速: 音韻連鎖
    else if (i < third * 2) alpha[i] = 0.30 + rng() * 0.18; // 中速: 語根
    else alpha[i] = 0.08 + rng() * 0.08;                    // 低速: 複合文脈・作業記憶
  }

  // 出力読み出し W_out (50 x FEAT_DIM) + モメンタム
  const W_out = new Float32Array(VOCAB_SIZE * FEAT_DIM);
  const V_out = new Float32Array(VOCAB_SIZE * FEAT_DIM);

  // 入力の既知度（音の並びの馴染み）を純ニューラルに測るための軽量音韻予測ヘッド W_ear (50 x (H + 51))
  // ※単語辞書ではなく、50音トークンの次音予測モデルそのもの
  const EAR_DIM = H + VOCAB_SIZE + 1;
  const W_ear = new Float32Array(VOCAB_SIZE * EAR_DIM);

  const h = new Float32Array(H);
  const hNext = new Float32Array(H);
  const feat = new Float32Array(FEAT_DIM);
  const earFeat = new Float32Array(EAR_DIM);
  const logits = new Float32Array(VOCAB_SIZE);
  const probs = new Float32Array(VOCAB_SIZE);
  const earProbs = new Float32Array(VOCAB_SIZE);
  const kokoro = new Float32Array([0.2, 0.2, 0.1, 0.2, 0.5]);

  let currentSurprise = 0.0;
  let totalTokensHeard = 0;

  function resetState() {
    h.fill(0);
    currentSurprise = 0.0;
  }

  function setKokoro(arr) {
    for (let i = 0; i < NUM_EMO; i++) {
      kokoro[i] = Math.max(0, Math.min(1, arr[i] ?? 0));
    }
  }

  function forwardEar() {
    let maxL = -Infinity;
    for (let v = 0; v < VOCAB_SIZE; v++) {
      const row = v * EAR_DIM;
      let s = 0;
      for (let k = 0; k < EAR_DIM; k++) {
        const fk = earFeat[k];
        if (fk !== 0) s += W_ear[row + k] * fk;
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
    for (let e = 0; e < NUM_EMO; e++) {
      feat[H + VOCAB_SIZE + e] = kokoro[e];
    }
    feat[H + IDX_SURPRISE] = surpriseVal;
    feat[FEAT_DIM - 1] = 1.0;

    earFeat.fill(0);
    earFeat.set(h, 0);
    earFeat[H + tokenId] = 1.0;
    earFeat[EAR_DIM - 1] = 1.0;
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
    for (let v = 0; v < VOCAB_SIZE; v++) probs[v] /= sumExp;
  }

  /**
   * [入力50音列, 目標応答50音列] を1音ずつ流し、
   * 1) 耳ヘッド(W_ear)で入力の馴染み（既知音韻パターン）を学び、
   * 2) 発話ヘッド(W_out)で応答50音列をオンライン学習する。
   */
  function trainPair50(inIds, outIds, opts = {}) {
    const { lr = 0.035, isUnknownEpisode = false } = opts;
    resetState();
    const momentum = 0.55;
    const wd = 0.0002;

    // 1) 入力フェーズ: 1音ずつ耳に入れつつ、既知エピソードなら耳ヘッド(W_ear)を訓練
    const surpriseLevel = isUnknownEpisode ? 1.0 : 0.0;
    for (let t = 0; t < inIds.length; t++) {
      const curTok = inIds[t];
      stepReservoir(curTok, surpriseLevel);
      totalTokensHeard += 1;

      if (!isUnknownEpisode && t + 1 < inIds.length) {
        const nextTok = inIds[t + 1];
        forwardEar();
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

    // 2) 応答フェーズ: 最後の入力状態から、1音ずつ目標応答の予測誤差で W_out を更新
    let loss = 0;
    let count = 0;
    for (let t = 0; t < outIds.length; t++) {
      const targetTok = outIds[t];
      forwardReadout();
      loss += -Math.log(Math.max(1e-7, probs[targetTok]));
      count += 1;

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

      // 次のステップへ進む（教師強制）
      if (t + 1 < outIds.length) {
        stepReservoir(targetTok, surpriseLevel * 0.8);
      }
    }

    return count > 0 ? loss / count : 0;
  }

  /**
   * 入力50音を1音ずつ聴取して自前の耳ヘッド(W_ear)で「予測誤差（驚き）」を計測し、
   * その驚きニューロンの活性込みで50音出力ニューロンを1音ずつ自己回帰発火する！
   */
  function respond50(promptIds, opts = {}) {
    const { maxLen = 14, mood = null } = opts;
    resetState();
    if (mood) setKokoro(mood);

    // 1) 入力聴取＋リアルタイム予測誤差（驚き）計測
    let sumNll = 0;
    let nllCount = 0;
    for (let t = 0; t < promptIds.length; t++) {
      const curTok = promptIds[t];
      // 1音前の耳予測と今の音を比べて驚きを更新
      if (t > 0 && curTok !== TOKEN_EOS) {
        const p = Math.max(1e-5, earProbs[curTok]);
        const nll = -Math.log(p); // 0(完全既知)〜約4(完全未知)
        sumNll += nll;
        nllCount += 1;
      }
      // ここまでの平均驚きを 0..1 にシグモイド圧縮して驚きニューロンへ入力
      const avgNll = nllCount > 0 ? sumNll / nllCount : 0;
      currentSurprise = 1 / (1 + Math.exp(-(avgNll - 2.1) * 3.0));

      stepReservoir(curTok, currentSurprise);
      forwardEar();
    }

    // 2) 50個の出力ニューロンを1音ずつ発火（if文の定型返答は一切なし！）
    const out = [];
    for (let s = 0; s < maxLen; s++) {
      forwardReadout();
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

  function respondText(yamatoText, opts = {}) {
    const inIds = encode50(yamatoText, { appendEos: true });
    const { outIds, surprise } = respond50(inIds, opts);
    return { reply: decode50(outIds), surprise };
  }

  return {
    name,
    resetState,
    setKokoro,
    getTotalTokensHeard: () => totalTokensHeard,
    getParamCount: () => W_in.length + recParamCount + W_out.length + W_ear.length,
    trainPair50,
    respond50,
    respondText,
  };
}
