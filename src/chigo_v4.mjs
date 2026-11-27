// chigo_v4.mjs — 非LLM・2系統シナプス＋脳内未来予測＋ワンショット学習 原始AGI「ちご v4」
// v3からの進化点:
//   1. 2系統シナプス:
//      - W_assoc  (句内の属性・共起シナプス: 「あめ・くる」「くま・こわ」「さむ・ひ」)
//      - W_causal (ターン跨ぎ時間差STDPシナプス: 「あめ」→「ひ・きえ」「くま」→「にげ」)
//   2. 脳内2手先シミュレーション (Mental Rollout):
//      入力から「1手先の未来(予測)」→「2手先の未来(対処)」をロールアウトし、
//      予測と対処を二語文で繋げて発話する。
//   3. 未知語への好奇心発火＋ワンショット可塑性:
//      シナプスが真っ白(0)な未知の1〜2文字語を聞いたら動的にニューロンを増設して
//      「〇〇・なに。」と問い返し、次の返答をワンショットで刻んで直後から2ホップ推論する。
// 入出力は codec50.mjs の50音ID列 (0..49) のみ、語彙は1〜2文字語に限定。

import { encode50, decode50, isValid50 } from './codec50.mjs';
import { NEURON_WORDS } from './neural_baby_v2.mjs';

const MAX_NEURONS = 64;
const INITIAL_THINGS = new Set([
  'ひ', 'き', 'み', 'みず', 'あめ', 'かぜ', 'やま', 'かわ',
  'くま', 'しか', 'うお', 'ひと', 'よ', 'ほし',
]);

export function createChigoV4(opts = {}) {
  const { name = 'ちごv4' } = opts;

  const words = [...NEURON_WORDS];
  const wordToIdx = new Map(words.map((w, i) => [w, i]));
  const isThing = new Uint8Array(MAX_NEURONS);
  for (let i = 0; i < words.length; i++) {
    isThing[i] = INITIAL_THINGS.has(words[i]) ? 1 : 0;
  }

  const IDX_NANI = wordToIdx.get('なに');
  const IDX_KURU = wordToIdx.get('くる');

  // 真っ白な初期シナプス（0.0 スタート: 未知語は厳密に総量0として検出可能）
  const W_assoc = new Float32Array(MAX_NEURONS * MAX_NEURONS);
  const W_causal = new Float32Array(MAX_NEURONS * MAX_NEURONS);

  let steps = 0;
  let prevTurnIds = [];
  let pendingUnknownIdx = -1;

  // 脳内ロールアウト状態ベクトル
  const stateNow = new Float32Array(MAX_NEURONS);
  const stateFuture1 = new Float32Array(MAX_NEURONS);
  const stateFuture2 = new Float32Array(MAX_NEURONS);

  function getOrCreateNeuron(word) {
    const clean = word.length <= 2 ? word : word.slice(0, 2);
    if (!clean) return -1;
    if (wordToIdx.has(clean)) return wordToIdx.get(clean);
    if (words.length >= MAX_NEURONS) return -1;
    const idx = words.length;
    words.push(clean);
    wordToIdx.set(clean, idx);
    isThing[idx] = 1;
    return idx;
  }

  function parseClauses(text, allowCreate = true) {
    const rawClauses = text.split('。').map((s) => s.trim()).filter(Boolean);
    const clauses = [];
    for (const clause of rawClauses) {
      const chunks = clause.split('・').map((s) => s.trim()).filter(Boolean);
      const ids = [];
      for (const ch of chunks) {
        if (wordToIdx.has(ch)) {
          ids.push(wordToIdx.get(ch));
          continue;
        }
        if (ch.length <= 2 && allowCreate) {
          const nid = getOrCreateNeuron(ch);
          if (nid >= 0) ids.push(nid);
          continue;
        }
        let i = 0;
        while (i < ch.length) {
          const two = ch.slice(i, i + 2);
          const one = ch.slice(i, i + 1);
          if (two.length === 2 && wordToIdx.has(two)) {
            ids.push(wordToIdx.get(two));
            i += 2;
          } else if (wordToIdx.has(one)) {
            ids.push(wordToIdx.get(one));
            i += 1;
          } else if (allowCreate) {
            const nid = getOrCreateNeuron(two);
            if (nid >= 0) ids.push(nid);
            i += two.length;
          } else {
            i += 1;
          }
        }
      }
      if (ids.length > 0) clauses.push(ids);
    }
    return clauses;
  }

  function normalizeRow(mat, pre, budget = 2.0) {
    const n = words.length;
    const row = pre * MAX_NEURONS;
    let sum = 0;
    for (let j = 0; j < n; j++) {
      let v = mat[row + j];
      if (v < 0.02) v = 0;
      mat[row + j] = v;
      sum += v;
    }
    if (sum > budget) {
      const scale = budget / sum;
      for (let j = 0; j < n; j++) mat[row + j] *= scale;
    }
  }

  function rowSynapseStrength(pre) {
    const n = words.length;
    const row = pre * MAX_NEURONS;
    let sum = 0;
    for (let j = 0; j < n; j++) {
      sum += W_assoc[row + j] + W_causal[row + j];
    }
    return sum;
  }

  function sparsePropagate(srcVec, mat, dstVec, topK = 3) {
    const n = words.length;
    dstVec.fill(0);
    for (let pre = 0; pre < n; pre++) {
      const a = srcVec[pre];
      if (a <= 0.15) continue;
      const row = pre * MAX_NEURONS;
      for (let post = 0; post < n; post++) {
        if (pre === post) continue;
        const w = mat[row + post];
        if (w > 0.02) dstVec[post] += a * w;
      }
    }
    const vals = [];
    for (let i = 0; i < n; i++) vals.push(dstVec[i]);
    vals.sort((a, b) => b - a);
    const thresh = vals[Math.min(topK - 1, vals.length - 1)] || 0;
    for (let i = 0; i < n; i++) {
      if (dstVec[i] < thresh || dstVec[i] <= 0) dstVec[i] = 0;
      else dstVec[i] = Math.tanh(dstVec[i]);
    }
  }

  function getStage() {
    if (steps < 12) return 0;
    if (steps < 28) return 1;
    return 2;
  }

  function perceive50(inputIds, lr = 0.22, opts = {}) {
    const { isNewContext = false, isTeaching = false } = opts;
    if (!isValid50(inputIds)) return { surprise: 0, unknownIdx: -1 };

    const text = decode50(inputIds);
    const clauses = parseClauses(text, true);
    const allIds = clauses.flat();
    const n = words.length;

    // 0) 未知語への「教え」を受け取った場合 → ワンショット刻印（エピソード→シナプス転写）
    let learnedUnknownNow = -1;
    if (lr > 0 && pendingUnknownIdx >= 0 && allIds.length > 0) {
      const target = pendingUnknownIdx;
      const row = target * MAX_NEURONS;
      for (const tid of allIds) {
        if (tid === target || tid === IDX_NANI) continue;
        W_assoc[row + tid] = Math.max(W_assoc[row + tid], 0.95);
        W_causal[row + tid] = Math.max(W_causal[row + tid], 0.90);
      }
      normalizeRow(W_assoc, target, 2.2);
      normalizeRow(W_causal, target, 2.2);
      learnedUnknownNow = target;
      pendingUnknownIdx = -1;
    }

    // 1) 未知語チェック（教えの直後は除く）
    let detectedUnknown = -1;
    if (learnedUnknownNow < 0) {
      for (const id of allIds) {
        if (id !== IDX_NANI && rowSynapseStrength(id) < 0.05) {
          detectedUnknown = id;
          break;
        }
      }
    }

    // 2) 2系統シナプスの学習
    if (lr > 0 && clauses.length > 0 && detectedUnknown < 0 && learnedUnknownNow < 0) {
      const touchedAssoc = new Set();
      const touchedCausal = new Set();

      // 2a) W_assoc: 句内の共起 (LTP + LTD)
      for (const clause of clauses) {
        const cSet = new Set(clause);
        for (let i = 0; i < clause.length; i++) {
          const pre = clause[i];
          const row = pre * MAX_NEURONS;
          touchedAssoc.add(pre);
          for (let j = 0; j < n; j++) {
            if (!cSet.has(j) && W_assoc[row + j] > 0) {
              W_assoc[row + j] = Math.max(0, W_assoc[row + j] - lr * 0.14 * W_assoc[row + j]);
            }
          }
          for (let j = 0; j < clause.length; j++) {
            if (i === j) continue;
            const post = clause[j];
            const dir = i < j ? 1.0 : 0.35;
            W_assoc[row + post] = Math.min(1.5, W_assoc[row + post] + lr * dir * (1.2 - W_assoc[row + post]));
          }
        }
      }

      // 2b) W_causal: 同一発話内の2句間（「くま・こわ。ひと・にげ。」→ くま→にげ）
      if (clauses.length >= 2) {
        const causeId = clauses[0][0];
        const effIds = clauses[1];
        const row = causeId * MAX_NEURONS;
        touchedCausal.add(causeId);
        for (const eff of effIds) {
          if (causeId === eff) continue;
          W_causal[row + eff] = Math.min(1.5, W_causal[row + eff] + lr * 0.85 * (1.2 - W_causal[row + eff]));
        }
      }

      // 2c) W_causal: ターン跨ぎ時間差STDP（直前の発話/出来事 → 今回の親の反応）
      if (!isNewContext && prevTurnIds.length > 0) {
        const curSet = new Set(allIds);
        for (const pre of prevTurnIds) {
          if (pre === IDX_KURU) continue;
          const row = pre * MAX_NEURONS;
          touchedCausal.add(pre);
          for (let j = 0; j < n; j++) {
            if (!curSet.has(j) && W_causal[row + j] > 0) {
              W_causal[row + j] = Math.max(0, W_causal[row + j] - lr * 0.10 * W_causal[row + j]);
            }
          }
          for (const post of allIds) {
            if (pre === post) continue;
            W_causal[row + post] = Math.min(1.5, W_causal[row + post] + lr * 0.75 * (1.2 - W_causal[row + post]));
          }
        }
      }

      for (const pre of touchedAssoc) normalizeRow(W_assoc, pre, 2.0);
      for (const pre of touchedCausal) normalizeRow(W_causal, pre, 2.2);

      prevTurnIds = isTeaching ? [] : clauses[0].slice(0, 2);
      steps += 1;
    }

    // 3) 脳内2手先シミュレーション (Mental Rollout)
    stateNow.fill(0);
    if (learnedUnknownNow >= 0) {
      stateNow[learnedUnknownNow] = 1.0;
    } else {
      for (const id of allIds) stateNow[id] = 1.0;
    }

    const tmpCausal1 = new Float32Array(MAX_NEURONS);
    const tmpAssoc1 = new Float32Array(MAX_NEURONS);
    sparsePropagate(stateNow, W_causal, tmpCausal1, 3);
    sparsePropagate(stateNow, W_assoc, tmpAssoc1, 3);

    for (let i = 0; i < n; i++) {
      // 句内の強い属性 (W_assoc) とターン跨ぎ因果 (W_causal) をバランスよく統合
      stateFuture1[i] = Math.tanh(tmpCausal1[i] * 0.95 + tmpAssoc1[i] * 0.85);
    }

    // 2手先の未来（2ホップ推論）: stateFuture1 からさらに1ホップ伝播！
    const tmpC2 = new Float32Array(MAX_NEURONS);
    const tmpA2 = new Float32Array(MAX_NEURONS);
    sparsePropagate(stateFuture1, W_causal, tmpC2, 3);
    sparsePropagate(stateFuture1, W_assoc, tmpA2, 3);
    for (let i = 0; i < n; i++) {
      stateFuture2[i] = Math.tanh(tmpC2[i] * 0.95 + tmpA2[i] * 0.85);
    }

    if (detectedUnknown >= 0) {
      pendingUnknownIdx = detectedUnknown;
    }

    return { surprise: detectedUnknown >= 0 ? 1.0 : 0.1, unknownIdx: detectedUnknown };
  }

  function speak50() {
    const n = words.length;
    const stage = getStage();

    if (pendingUnknownIdx >= 0) {
      const unkWord = words[pendingUnknownIdx];
      const msg = stage === 0 ? `${unkWord}。なに。` : `${unkWord}・なに。`;
      return encode50(msg, { appendEos: true });
    }

    const pickBest = (vec, wantThing, excludeSet = new Set()) => {
      let bestIdx = -1;
      let bestVal = -Infinity;
      for (let i = 0; i < n; i++) {
        if (excludeSet.has(i) || i === IDX_NANI) continue;
        if (wantThing !== null && Boolean(isThing[i]) !== wantThing) continue;
        if (vec[i] > bestVal) {
          bestVal = vec[i];
          bestIdx = i;
        }
      }
      return { idx: bestIdx, val: bestVal };
    };

    const used = new Set();
    const clauses = [];

    const curSubj = pickBest(stateNow, true, used);
    const curPred = pickBest(stateNow, false, used);

    if (stage === 0) {
      const top = pickBest(stateFuture1, null, used);
      const word = top.val > 0.15 ? words[top.idx] : (curSubj.idx >= 0 ? words[curSubj.idx] : 'う');
      clauses.push(word);
    } else {
      // 1句目: 1手先の未来予測
      const f1Thing = pickBest(stateFuture1, true, new Set([curSubj.idx]));
      const f1Pred = pickBest(stateFuture1, false, new Set([IDX_KURU]));

      // 入力が状態語だけ（例: 「さむ」）なら、1手先の対象（例: 「ひ」）と組ませる
      if (curSubj.idx < 0 && curPred.idx >= 0 && f1Thing.val > 0.2) {
        clauses.push(`${words[f1Thing.idx]}・${words[curPred.idx]}`);
        used.add(f1Thing.idx);
        used.add(curPred.idx);
      } else {
        const subj1 = f1Thing.val > 0.45 ? f1Thing.idx : (curSubj.idx >= 0 ? curSubj.idx : f1Thing.idx);
        const pred1 = f1Pred.val > 0.15 ? f1Pred.idx : (curPred.idx >= 0 ? curPred.idx : -1);
        if (subj1 >= 0 && pred1 >= 0) {
          clauses.push(`${words[subj1]}・${words[pred1]}`);
          used.add(pred1);
        } else if (subj1 >= 0) {
          clauses.push(words[subj1]);
        }
      }

      // 2句目 (Stage 2): 2手先の未来（2ホップ推論！）
      if (stage >= 2) {
        const f2Thing = pickBest(stateFuture2, true, new Set([curSubj.idx]));
        const f2Pred = pickBest(stateFuture2, false, new Set([...used, IDX_KURU]));
        const anchorThing = f2Thing.val > 0.25 ? f2Thing.idx : (f1Thing.val > 0.25 ? f1Thing.idx : curSubj.idx);

        if (anchorThing >= 0 && f2Pred.val > 0.22) {
          clauses.push(`${words[anchorThing]}・${words[f2Pred.idx]}`);
        } else if (f2Thing.val > 0.3 && f1Pred.val > 0.2 && !used.has(f2Thing.idx)) {
          clauses.push(`${words[f2Thing.idx]}・${words[f1Pred.idx]}`);
        }
      }
    }

    if (clauses.length === 0) clauses.push('なに');
    return encode50(clauses.join('。') + '。', { appendEos: true });
  }

  function probeText(yamatoText) {
    pendingUnknownIdx = -1;
    const inIds = encode50(yamatoText, { appendEos: true });
    perceive50(inIds, 0.0, { isNewContext: true });
    return decode50(speak50());
  }

  function talkText(yamatoText) {
    const inIds = encode50(yamatoText, { appendEos: true });
    perceive50(inIds, 0.25, { isNewContext: true, isTeaching: pendingUnknownIdx >= 0 });
    return decode50(speak50());
  }

  function getDensity() {
    const n = words.length;
    let active = 0;
    for (let i = 0; i < n; i++) {
      const row = i * MAX_NEURONS;
      for (let j = 0; j < n; j++) {
        if (W_assoc[row + j] > 0.05 || W_causal[row + j] > 0.05) active += 1;
      }
    }
    return Number(((active / (n * n)) * 100).toFixed(1));
  }

  return {
    name,
    getStage,
    getSteps: () => steps,
    getVocabSize: () => words.length,
    getDensity,
    perceive50,
    speak50,
    probeText,
    talkText,
  };
}
