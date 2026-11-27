// chigo_hito_v10.mjs — 純ニューラル「かなで話す人」ちご v10
// 100%純ニューラル・ゼロ依存・厳密50音I/O (0..49)
// 単語辞書・Map・DP分節・if文による定型返答は一切ゼロ！
//
// v9からの大進化（人間と「かな」で自然に対話できる自律個体へ）:
//   1. 自律「こころ」ニューラル回路 (W_emo: 5 × (H + 1)):
//      v9では外部から setKokoro() で毎回注入していた情動5軸を、
//      聞いた50音列のリザバー状態 h から自力で推論・更新する。
//      1ターン目に「さむ」と聞けば自分の心が「さむ」へ傾き、
//      2ターン目に「あめ」と聞けば外部ヒントなしで「ひ・きえ。」へ分岐する！
//   2. ターン跨ぎ会話文脈トレース (slowCtx):
//      発話終了時にリザバーの低速帯域を会話文脈として穏やかに引き継ぎ、
//      複数ターンの会話の流れを純ニューラル空間で保持する。
//   3. オンライン未知語習得＋訂正学習＋睡眠定着 (W_fast + 海馬リプレイ):
//      未知語には驚き発火で「なに。」と聞き返し、次の返答で即座に高速シナプスへ刻む。
//      既存の返答を人間が教え直す（訂正）ことも可能で、睡眠リプレイで長期皮質へ定着する。
//   4. 脳の保存・復元 (serialize / deserialize):
//      育てたシナプス重み・海馬・こころを JSON として永続化できる。

import { VOCAB_SIZE, TOKEN_EOS, encode50, decode50 } from './codec50.mjs';
import { createRng } from './jomon_world.mjs';

export const EMO_NAMES = ['さむ', 'はら', 'こわ', 'きに', 'やす'];
const NUM_EMO = EMO_NAMES.length;
const IN_DIM = VOCAB_SIZE + NUM_EMO + 1; // 50音 + こころ5軸 + 驚き1軸 = 56
const IDX_SURPRISE = VOCAB_SIZE + NUM_EMO;

function sigmoid(x) {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

/**
 * 初期知識ゼロの純ニューラル「かなで話す人」ちご v10 を生成する。
 */
export function createKanaHitoV10(opts = {}) {
  const { name = 'ちごv10(かなで話す人)', hiddenSize = 256, seed = 108 } = opts;
  const rng = createRng(seed);
  const H = hiddenSize;
  const third = Math.floor(H / 3);
  const SLOW_START = third * 2;
  const FEAT_DIM = H + IN_DIM + 1;
  const EAR_DIM = H + VOCAB_SIZE + 1;
  const EMO_DIM = SLOW_START + 1;

  // 入力結合 W_in
  const W_in = new Float32Array(H * IN_DIM);
  for (let i = 0; i < W_in.length; i++) {
    W_in[i] = (rng() * 2 - 1) * 0.48;
  }

  // スパース再帰結合 W_rec
  // 聴覚皮質 (0..SLOW_START-1) は情動入力から独立させ、50音列そのものから情動を自力推論できるようにする
  const recIndices = [];
  const recWeights = [];
  for (let i = 0; i < H; i++) {
    const idxs = [];
    const wts = [];
    const maxJ = i < SLOW_START ? SLOW_START : H;
    for (let j = 0; j < maxJ; j++) {
      if (rng() < 0.13) {
        idxs.push(j);
        wts.push((rng() * 2 - 1) * 0.2);
      }
    }
    recIndices.push(Int32Array.from(idxs));
    recWeights.push(Float32Array.from(wts));
  }

  // 3時間スケール漏れ率 (速・中・遅)
  const alpha = new Float32Array(H);
  for (let i = 0; i < H; i++) {
    if (i < third) alpha[i] = 0.78 + rng() * 0.18;
    else if (i < SLOW_START) alpha[i] = 0.30 + rng() * 0.18;
    else alpha[i] = 0.08 + rng() * 0.08;
  }

  // 長期皮質シナプス (slow cortex)
  const W_out = new Float32Array(VOCAB_SIZE * FEAT_DIM);
  const V_out = new Float32Array(VOCAB_SIZE * FEAT_DIM);
  const W_ear = new Float32Array(VOCAB_SIZE * EAR_DIM);

  // 自律「こころ」情動読み出しシナプス (5 × (SLOW_START + 1))
  const W_emo = new Float32Array(NUM_EMO * EMO_DIM);

  // 高速シナプス (fast weights: リザバー状態 h のみに局所結合)
  const W_fast = new Float32Array(VOCAB_SIZE * H);
  const W_ear_fast = new Float32Array(VOCAB_SIZE * H);

  // 海馬エピソード記憶バッファ（生の50音ID列と心の状態のみ）
  const hippocampus = [];
  const MAX_HIPPO = 96;

  let pendingNovelIds = null;
  let pendingNovelMood = null;
  let lastInIds = null;
  let lastMood = null;

  const h = new Float32Array(H);
  const hNext = new Float32Array(H);
  const slowCtx = new Float32Array(H - SLOW_START);
  const feat = new Float32Array(FEAT_DIM);
  const earFeat = new Float32Array(EAR_DIM);
  const logits = new Float32Array(VOCAB_SIZE);
  const probs = new Float32Array(VOCAB_SIZE);
  const earProbs = new Float32Array(VOCAB_SIZE);
  const emoPred = new Float32Array(NUM_EMO);
  const kokoro = new Float32Array([0.2, 0.2, 0.1, 0.3, 0.6]);

  let currentSurprise = 0.0;
  let turnCount = 0;
  let sleepCount = 0;

  function resetState(clearContext = false) {
    h.fill(0);
    currentSurprise = 0.0;
    if (clearContext) {
      slowCtx.fill(0);
      kokoro.set([0.2, 0.2, 0.1, 0.3, 0.6]);
      pendingNovelIds = null;
      pendingNovelMood = null;
    }
  }

  function applySlowContext(gain = 0.15) {
    for (let i = SLOW_START; i < H; i++) {
      h[i] = slowCtx[i - SLOW_START] * gain;
    }
  }

  function captureSlowContext(decay = 0.65) {
    for (let i = SLOW_START; i < H; i++) {
      const idx = i - SLOW_START;
      slowCtx[idx] = slowCtx[idx] * (1 - decay) + h[i] * decay;
    }
  }

  function setKokoro(arr) {
    for (let i = 0; i < NUM_EMO; i++) {
      kokoro[i] = Math.max(0, Math.min(1, arr[i] ?? 0));
    }
  }

  function getKokoro() {
    return Array.from(kokoro, (v) => Number(v.toFixed(3)));
  }

  function recordHippo(inIds, outIds, mood, isNew = false) {
    const key = Array.from(inIds).join(',');
    const existingIdx = hippocampus.findIndex(
      (e) =>
        Array.from(e.inIds).join(',') === key &&
        Math.abs(e.mood[0] - mood[0]) < 0.25 &&
        Math.abs(e.mood[1] - mood[1]) < 0.25 &&
        Math.abs(e.mood[2] - mood[2]) < 0.25
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

  function syncFeatures(tokenId, surpriseVal) {
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

  function stepReservoir(tokenId, surpriseVal = currentSurprise) {
    for (let i = 0; i < H; i++) {
      const inRow = i * IN_DIM;
      let sum = W_in[inRow + tokenId] + W_in[inRow + IDX_SURPRISE] * surpriseVal;
      if (i >= SLOW_START) {
        for (let e = 0; e < NUM_EMO; e++) {
          sum += W_in[inRow + VOCAB_SIZE + e] * kokoro[e] * 0.35;
        }
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
    syncFeatures(tokenId, surpriseVal);
  }

  function forwardEmo() {
    for (let e = 0; e < NUM_EMO; e++) {
      const row = e * EMO_DIM;
      let s = W_emo[row + SLOW_START]; // bias
      for (let k = 0; k < SLOW_START; k++) {
        s += W_emo[row + k] * h[k];
      }
      emoPred[e] = sigmoid(s);
    }
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

  /**
   * 長期皮質＋耳予測＋自律こころ回路を同時に訓練する。
   */
  function trainCortexPair50(inIds, outIds, opts = {}) {
    const {
      lr = 0.032,
      isUnknownEpisode = false,
      saveToHippo = true,
      trainEmo = true,
      contextJitter = 0.05,
    } = opts;
    resetState(false);

    if (contextJitter > 0) {
      for (let i = SLOW_START; i < H; i++) {
        h[i] = (rng() * 2 - 1) * contextJitter;
      }
    }

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

    // 自律「こころ」回路 W_emo の学習（聴覚皮質 0..SLOW_START-1 のみから情動5軸を推論）
    if (trainEmo && !isUnknownEpisode) {
      forwardEmo();
      const emoLr = lr * 5.0;
      for (let e = 0; e < NUM_EMO; e++) {
        const err = emoPred[e] - kokoro[e];
        if (Math.abs(err) < 1e-4) continue;
        const row = e * EMO_DIM;
        for (let k = 0; k < SLOW_START; k++) {
          W_emo[row + k] -= emoLr * (err * h[k] + wd * W_emo[row + k]);
        }
        W_emo[row + SLOW_START] -= emoLr * err;
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
   * 昼のオンライン即席記憶・訂正学習 (高速シナプス W_fast / W_ear_fast)
   */
  function imprintFast50(inIds, outIds, mood, stepsCount = 16) {
    recordHippo(inIds, outIds, mood, true);
    const daytimeNew = hippocampus.filter((e) => e.isNew);
    const anchorSample = hippocampus.filter((e) => !e.isNew).slice(0, 8);

    const lr = 0.075;
    const wd = 0.006;
    for (let rep = 0; rep < stepsCount; rep++) {
      const batch = [...daytimeNew, ...anchorSample];
      for (const item of batch) {
        setKokoro(item.mood);
        resetState(false);
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
   * 夜の睡眠リプレイ定着:
   * 海馬にある新エピソード＋既存エピソードを交互リプレイして長期皮質へ定着させ、
   * 高速シナプス W_fast を完全 0 消去する。
   */
  function sleepConsolidate(epochs = 26) {
    const NANI_IDS = encode50('なに。', { appendEos: true });
    for (let ep = 0; ep < epochs; ep++) {
      const lr = ep < 14 ? 0.032 : 0.018;
      for (const e of hippocampus) {
        setKokoro(e.mood);
        const repeats = e.isNew ? 2 : 1;
        for (let r = 0; r < repeats; r++) {
          trainCortexPair50(e.inIds, e.outIds, {
            lr,
            isUnknownEpisode: false,
            saveToHippo: false,
            trainEmo: true,
          });
        }
      }
      for (let n = 0; n < 2; n++) {
        const noise = new Uint8Array([
          4 + Math.floor(rng() * 46),
          4 + Math.floor(rng() * 46),
          TOKEN_EOS,
        ]);
        setKokoro([0.2, 0.2, 0.1, 0.8, 0.3]);
        trainCortexPair50(noise, NANI_IDS, {
          lr: lr * 0.7,
          isUnknownEpisode: true,
          saveToHippo: false,
          trainEmo: false,
        });
      }
    }

    W_fast.fill(0);
    W_ear_fast.fill(0);
    for (const e of hippocampus) e.isNew = false;
    sleepCount += 1;
    resetState(true);
  }

  /**
   * 純ニューラル応答（自律こころ更新＋ターン跨ぎ文脈対応）
   */
  function respond50(promptIds, opts = {}) {
    const {
      maxLen = 14,
      mood = null,
      recordPending = false,
      autoKokoro = true,
      keepContext = true,
    } = opts;

    resetState(false);
    if (mood) setKokoro(mood);
    if (keepContext) applySlowContext(0.18);

    let sumNll = 0;
    let nllCount = 0;
    const lastTok = promptIds.length > 0 ? promptIds[promptIds.length - 1] : TOKEN_EOS;

    for (let t = 0; t < promptIds.length; t++) {
      const curTok = promptIds[t];
      if (t > 0 && curTok !== TOKEN_EOS) {
        const p = Math.max(1e-5, earProbs[curTok]);
        sumNll += -Math.log(p);
        nllCount += 1;
      }
      const avgNll = nllCount > 0 ? sumNll / nllCount : 0;
      currentSurprise = 1 / (1 + Math.exp(-(avgNll - 2.55) * 3.0));
      stepReservoir(curTok, currentSurprise);
      forwardEar(true);
    }

    // 自律「こころ」更新: 外部から mood が指定されていない場合、聴取後の h から情動を自力推論して更新
    if (autoKokoro && !mood) {
      forwardEmo();
      if (currentSurprise > 0.65) {
        // 未知語への驚き時は好奇心（きに）を自然に高める
        kokoro[3] = Math.min(1, kokoro[3] * 0.4 + 0.85 * 0.6);
      } else {
        // 情動の強さ（0.5からの離れ具合）が大きい軸ほど強く心を揺らす（曖昧語は直前の心を維持）
        for (let e = 0; e < NUM_EMO; e++) {
          const strength = Math.abs(emoPred[e] - 0.5) * 2; // 0..1
          const blend = 0.08 + 0.84 * (strength * strength);
          kokoro[e] = (1 - blend) * kokoro[e] + blend * emoPred[e];
        }
      }
      syncFeatures(lastTok, currentSurprise);
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

    if (keepContext) captureSlowContext(0.65);

    lastInIds = Uint8Array.from(promptIds);
    lastMood = Float32Array.from(kokoro);
    turnCount += 1;

    return {
      outIds: Uint8Array.from(out),
      surprise: Number(currentSurprise.toFixed(2)),
      kokoro: getKokoro(),
    };
  }

  function interact50(promptIds, opts = {}) {
    const {
      maxLen = 14,
      mood = null,
      allowOnlineLearn = true,
      autoKokoro = true,
      keepContext = true,
    } = opts;
    if (mood) setKokoro(mood);

    // 直前ターンで未知語に「なに。」と聞き返していたら、今回の発言を教えとして即座に高速記憶
    if (allowOnlineLearn && pendingNovelIds) {
      const targetIn = pendingNovelIds;
      const targetMood = pendingNovelMood;
      pendingNovelIds = null;
      pendingNovelMood = null;
      imprintFast50(targetIn, promptIds, targetMood, 16);
      const res = respond50(targetIn, {
        maxLen,
        mood: targetMood,
        recordPending: false,
        autoKokoro: false,
        keepContext,
      });
      return { ...res, learned: true };
    }

    const res = respond50(promptIds, {
      maxLen,
      mood,
      recordPending: allowOnlineLearn,
      autoKokoro,
      keepContext,
    });
    return { ...res, learned: false };
  }

  function talkText(kanaText, opts = {}) {
    const inIds = encode50(kanaText, { appendEos: true });
    const { outIds, surprise, kokoro: k, learned } = interact50(inIds, opts);
    return { reply: decode50(outIds), surprise, kokoro: k, learned };
  }

  function probeText(kanaText, opts = {}) {
    const inIds = encode50(kanaText, { appendEos: true });
    const { outIds, surprise, kokoro: k } = respond50(inIds, {
      ...opts,
      recordPending: false,
    });
    return { reply: decode50(outIds), surprise, kokoro: k };
  }

  /**
   * 任意の「ことば => へんじ」をオンライン教示（訂正学習にも使える）
   */
  function teachText(inText, outText, opts = {}) {
    const inIds = encode50(inText, { appendEos: true });
    const outIds = encode50(outText, { appendEos: true });
    const targetMood = opts.mood ? Float32Array.from(opts.mood) : Float32Array.from(kokoro);
    imprintFast50(inIds, outIds, targetMood, opts.stepsCount ?? 18);
    pendingNovelIds = null;
    pendingNovelMood = null;
    return probeText(inText, { mood: targetMood, autoKokoro: false, keepContext: false });
  }

  /**
   * 直前の発言に対する返答を訂正する
   */
  function correctLastText(betterReplyText) {
    if (!lastInIds) return null;
    const outIds = encode50(betterReplyText, { appendEos: true });
    const targetMood = lastMood ? Float32Array.from(lastMood) : Float32Array.from(kokoro);
    imprintFast50(lastInIds, outIds, targetMood, 18);
    pendingNovelIds = null;
    pendingNovelMood = null;
    const { outIds: verified, surprise, kokoro: k } = respond50(lastInIds, {
      mood: targetMood,
      recordPending: false,
      autoKokoro: false,
      keepContext: false,
    });
    return { reply: decode50(verified), surprise, kokoro: k };
  }

  function getFastNorm() {
    let s = 0;
    for (let i = 0; i < W_fast.length; i++) s += W_fast[i] * W_fast[i];
    return Number(Math.sqrt(s).toFixed(3));
  }

  function serialize() {
    return {
      version: 10,
      name,
      hiddenSize: H,
      seed,
      turnCount,
      sleepCount,
      W_in: Array.from(W_in),
      recIndices: recIndices.map((a) => Array.from(a)),
      recWeights: recWeights.map((a) => Array.from(a)),
      alpha: Array.from(alpha),
      W_out: Array.from(W_out),
      V_out: Array.from(V_out),
      W_ear: Array.from(W_ear),
      W_emo: Array.from(W_emo),
      W_fast: Array.from(W_fast),
      W_ear_fast: Array.from(W_ear_fast),
      kokoro: Array.from(kokoro),
      slowCtx: Array.from(slowCtx),
      hippocampus: hippocampus.map((e) => ({
        inIds: Array.from(e.inIds),
        outIds: Array.from(e.outIds),
        mood: Array.from(e.mood),
        isNew: e.isNew,
      })),
    };
  }

  function deserialize(data) {
    if (!data || data.version !== 10 || data.hiddenSize !== H) {
      throw new Error('Incompatible brain snapshot for chigo v10');
    }
    W_in.set(data.W_in);
    for (let i = 0; i < H; i++) {
      recIndices[i] = Int32Array.from(data.recIndices[i]);
      recWeights[i] = Float32Array.from(data.recWeights[i]);
    }
    alpha.set(data.alpha);
    W_out.set(data.W_out);
    V_out.set(data.V_out);
    W_ear.set(data.W_ear);
    W_emo.set(data.W_emo);
    W_fast.set(data.W_fast);
    W_ear_fast.set(data.W_ear_fast);
    kokoro.set(data.kokoro);
    slowCtx.set(data.slowCtx);
    hippocampus.length = 0;
    for (const e of data.hippocampus) {
      hippocampus.push({
        inIds: Uint8Array.from(e.inIds),
        outIds: Uint8Array.from(e.outIds),
        mood: Float32Array.from(e.mood),
        isNew: Boolean(e.isNew),
      });
    }
    turnCount = data.turnCount ?? 0;
    sleepCount = data.sleepCount ?? 0;
    pendingNovelIds = null;
    pendingNovelMood = null;
  }

  return {
    name,
    hiddenSize: H,
    resetState,
    setKokoro,
    getKokoro,
    getHippoSize: () => hippocampus.length,
    getFastNorm,
    getStats: () => ({
      name,
      hiddenSize: H,
      turnCount,
      sleepCount,
      hippoSize: hippocampus.length,
      fastNorm: getFastNorm(),
      kokoro: getKokoro(),
      awaitingTeach: Boolean(pendingNovelIds),
    }),
    trainCortexPair50,
    sleepConsolidate,
    talkText,
    probeText,
    teachText,
    correctLastText,
    serialize,
    deserialize,
  };
}

/**
 * 幼少期の「かな会話」育成コーパス（すべて50音のみ・脳の外側にある親の語りかけ）
 * こころ5軸: [さむ, はら, こわ, きに, やす]
 */
export const KANA_HITO_DIALOGUES = [
  // あいさつ・交流
  { in: 'おはよう',   out: 'おはよう。',     mood: [0.15, 0.2, 0.05, 0.4, 0.85] },
  { in: 'こんにちは', out: 'こんにちは。',   mood: [0.15, 0.2, 0.05, 0.4, 0.85] },
  { in: 'こんばんは', out: 'こんばんは。',   mood: [0.25, 0.2, 0.1, 0.3, 0.8] },
  { in: 'やあ',       out: 'うれしい。',     mood: [0.1, 0.15, 0.05, 0.5, 0.9] },
  { in: 'またね',     out: 'またね。',       mood: [0.15, 0.15, 0.05, 0.3, 0.85] },
  { in: 'おやすみ',   out: 'よ・ねる。',     mood: [0.2, 0.1, 0.05, 0.15, 0.95] },
  { in: 'ありがとう', out: 'うれしい。',     mood: [0.1, 0.1, 0.05, 0.4, 0.95] },
  { in: 'すき',       out: 'うれしい・すき。', mood: [0.1, 0.1, 0.05, 0.5, 0.95] },
  { in: 'なまえ',     out: 'ちご。',         mood: [0.15, 0.15, 0.05, 0.5, 0.85] },
  { in: 'だれ',       out: 'ちご・ひと。',   mood: [0.15, 0.15, 0.05, 0.5, 0.85] },
  { in: 'げんき',     out: 'げんき。',       mood: [0.1, 0.15, 0.05, 0.4, 0.9] },

  // 情動分岐（同じ「あめ」でも自分の心の状態で返答が変わる）
  { in: 'あめ',       out: 'ひ・きえ。',     mood: [0.95, 0.1, 0.1, 0.1, 0.1] },
  { in: 'あめ・くる', out: 'ひ・きえ。',     mood: [0.95, 0.1, 0.1, 0.1, 0.1] },
  { in: 'あめ',       out: 'みず・のむ。',   mood: [0.1, 0.95, 0.1, 0.1, 0.3] },

  // からだ・自然・サバイバル
  { in: 'さむ',       out: 'ひ・たす。',     mood: [0.95, 0.1, 0.1, 0.1, 0.1] },
  { in: 'さむい',     out: 'ひ・たす。',     mood: [0.95, 0.1, 0.1, 0.1, 0.1] },
  { in: 'はら',       out: 'み・くう。',     mood: [0.1, 0.95, 0.1, 0.1, 0.2] },
  { in: 'はら・へる', out: 'み・くう。',     mood: [0.1, 0.95, 0.1, 0.1, 0.2] },
  { in: 'みず',       out: 'みず・のむ。',   mood: [0.1, 0.9, 0.05, 0.15, 0.45] },
  { in: 'くま',       out: 'ひと・にげ。',   mood: [0.1, 0.1, 0.95, 0.1, 0.05] },
  { in: 'こわい',     out: 'ひと・にげ。',   mood: [0.1, 0.1, 0.95, 0.1, 0.05] },
  { in: 'しか',       out: 'しか・とる。',   mood: [0.1, 0.85, 0.1, 0.25, 0.3] },
  { in: 'よ',         out: 'よ・ねる。',     mood: [0.2, 0.1, 0.1, 0.2, 0.9] },
  { in: 'ねむい',     out: 'よ・ねる。',     mood: [0.2, 0.1, 0.05, 0.15, 0.95] },
  { in: 'ひ',         out: 'あたたか。',     mood: [0.2, 0.15, 0.05, 0.35, 0.85] },
  { in: 'そら',       out: 'ひろい。',       mood: [0.15, 0.15, 0.05, 0.65, 0.85] },
  { in: 'うみ',       out: 'みず・ひろい。', mood: [0.15, 0.2, 0.05, 0.65, 0.85] },
  { in: 'はな',       out: 'きれい。',       mood: [0.1, 0.1, 0.05, 0.6, 0.9] },
  { in: 'とり',       out: 'そら・とぶ。',   mood: [0.1, 0.1, 0.05, 0.65, 0.85] },

  // 複合文脈（ゼロショット汎化の足場）
  { in: 'よ・さむ',   out: 'ひ・たす。',     mood: [0.92, 0.1, 0.1, 0.1, 0.25] },
  { in: 'あめ・さむ', out: 'ひ・きえ。',     mood: [0.95, 0.1, 0.1, 0.1, 0.1] },
  { in: 'はら・みず', out: 'みず・のむ。',   mood: [0.1, 0.92, 0.05, 0.1, 0.4] },
  { in: 'はら・しか', out: 'しか・とる。',   mood: [0.1, 0.9, 0.1, 0.2, 0.25] },
  { in: 'みず・くま', out: 'ひと・にげ。',   mood: [0.1, 0.2, 0.95, 0.1, 0.05] },
];

/**
 * 初期知識ゼロのちご v10 を「かなで話す人」へ育てる幼少期訓練ループ
 */
export function raiseKanaHitoV10(hito, opts = {}) {
  const { epochs = 48, seed = 2026 } = opts;
  const rng = createRng(seed);
  const NANI_IDS = encode50('なに。', { appendEos: true });

  function randomNoise50() {
    const len = 2 + Math.floor(rng() * 2);
    const ids = new Uint8Array(len + 1);
    for (let i = 0; i < len; i++) ids[i] = 4 + Math.floor(rng() * (VOCAB_SIZE - 4));
    ids[len] = TOKEN_EOS;
    return ids;
  }

  for (let ep = 1; ep <= epochs; ep++) {
    const lr = ep < 22 ? 0.038 : 0.02;
    for (const d of KANA_HITO_DIALOGUES) {
      hito.setKokoro(d.mood);
      hito.trainCortexPair50(
        encode50(d.in, { appendEos: true }),
        encode50(d.out, { appendEos: true }),
        { lr, isUnknownEpisode: false, saveToHippo: ep === epochs, trainEmo: true }
      );
    }
    for (let n = 0; n < 3; n++) {
      hito.setKokoro([0.2, 0.2, 0.1, 0.85, 0.3]);
      hito.trainCortexPair50(randomNoise50(), NANI_IDS, {
        lr: lr * 0.85,
        isUnknownEpisode: true,
        saveToHippo: false,
        trainEmo: false,
      });
    }
  }
  hito.resetState(true);
  return hito;
}
