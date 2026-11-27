// chigo_v6.mjs — 「こころ（情動・欲求）」を持つ非LLM原始AGI「ちご v6」
// v5（初期辞書ゼロ・区切りなし連続50音からの語根自己発見＋2系統STDP）に、
// 生きた「こころ（5軸の内部情動状態）」と「言葉↔こころ双方向シナプス (W_emo)」を追加。
//
// 特徴:
//   1. 同じ言葉（例: 「あめ」）を聞いても、今の「こころ」の状態（寒い vs 渇いた）で
//      脳内の活性化が変わり、違う応答・行動提案が立ち上がる。
//   2. 外部入力ゼロ（静かな時間）でも、「こころ」の欲求・恐れ・好奇心から
//      自発的に1〜2文字のヤマトコトバをつぶやく (spontaneous50)。
//   3. 言葉が「こころ」を揺らし、「こころ」が言葉を選ぶ循環ループを持つ。

import { encode50, decode50, isValid50 } from './codec50.mjs';

const MAX_NEURONS = 96;

export const KOKORO_NAMES = ['さむ', 'はら', 'こわ', 'きに', 'やす'];
export const EMO_SAMU = 0;
export const EMO_HARA = 1;
export const EMO_KOWA = 2;
export const EMO_KINI = 3;
export const EMO_YASU = 4;
const NUM_EMO = 5;

function idsToMoraChars(ids) {
  const text = decode50(ids);
  const morae = [];
  for (const ch of text) {
    if (ch === '_' || ch === '・' || ch === '。') morae.push('。');
    else morae.push(ch);
  }
  return morae;
}

export function createChigoV6(opts = {}) {
  const { name = 'ちごv6' } = opts;

  const unigramCount = new Map();
  const bigramCount = new Map();
  let totalMorae = 0;

  const words = ['なに'];
  const wordToIdx = new Map([['なに', 0]]);
  const IDX_NANI = 0;

  const W_assoc = new Float32Array(MAX_NEURONS * MAX_NEURONS);
  const W_causal = new Float32Array(MAX_NEURONS * MAX_NEURONS);
  const W_emo = new Float32Array(MAX_NEURONS * NUM_EMO);

  const kokoro = new Float32Array([0.15, 0.20, 0.05, 0.35, 0.50]);

  const stateNow = new Float32Array(MAX_NEURONS);
  const stateFuture1 = new Float32Array(MAX_NEURONS);
  const stateFuture2 = new Float32Array(MAX_NEURONS);

  let steps = 0;
  let prevTurnIds = [];

  function birthNeuron(word) {
    if (!word || word.length < 1 || word.length > 2) return -1;
    if (wordToIdx.has(word)) return wordToIdx.get(word);
    if (words.length >= MAX_NEURONS) return -1;
    const idx = words.length;
    words.push(word);
    wordToIdx.set(word, idx);
    return idx;
  }

  function updatePhonemeStatsAndDiscover(morae) {
    for (let i = 0; i < morae.length; i++) {
      const c1 = morae[i];
      if (c1 === '。') continue;
      totalMorae += 1;
      unigramCount.set(c1, (unigramCount.get(c1) || 0) + 1);

      if (i + 1 < morae.length && morae[i + 1] !== '。') {
        const c2 = morae[i + 1];
        const bi = c1 + c2;
        const bc = (bigramCount.get(bi) || 0) + 1;
        bigramCount.set(bi, bc);

        if (bc >= 3 && !wordToIdx.has(bi)) {
          const c1Count = unigramCount.get(c1) || 1;
          const c2Count = unigramCount.get(c2) || 1;
          const tpFwd = bc / c1Count;
          const tpBwd = bc / c2Count;
          const pmi = Math.log2((bc * totalMorae) / (c1Count * c2Count));
          if ((tpFwd >= 0.32 || tpBwd >= 0.32) && pmi >= 1.2) {
            birthNeuron(bi);
          }
        }
      }

      const prevB = i === 0 || morae[i - 1] === '。';
      const nextB = i + 1 >= morae.length || morae[i + 1] === '。';
      if ((prevB && nextB) || (unigramCount.get(c1) >= 8 && prevB)) {
        if (!wordToIdx.has(c1)) {
          let maxFwd = 0;
          for (const [bi, bc] of bigramCount.entries()) {
            if (bi[0] === c1) {
              const tp = bc / (unigramCount.get(c1) || 1);
              if (tp > maxFwd) maxFwd = tp;
            }
          }
          if (maxFwd < 0.55) birthNeuron(c1);
        }
      }
    }
  }

  function segmentContinuous(morae, allowBirthUnknown = false) {
    const segments = [];
    let cur = [];
    for (const m of morae) {
      if (m === '。') {
        if (cur.length > 0) segments.push(cur);
        cur = [];
      } else {
        cur.push(m);
      }
    }
    if (cur.length > 0) segments.push(cur);

    const clauseIdsList = [];
    for (const seg of segments) {
      const L = seg.length;
      const dp = new Float64Array(L + 1).fill(-1e9);
      const back = new Int32Array(L + 1).fill(-1);
      dp[0] = 0;

      for (let i = 0; i < L; i++) {
        if (dp[i] < -1e8) continue;
        const w1 = seg[i];
        const s1 = wordToIdx.has(w1) ? 1.6 : 0.1;
        if (dp[i] + s1 > dp[i + 1]) {
          dp[i + 1] = dp[i] + s1;
          back[i + 1] = i;
        }
        if (i + 2 <= L) {
          const w2 = seg[i] + seg[i + 1];
          let s2 = 0.0;
          if (wordToIdx.has(w2)) {
            const bc = bigramCount.get(w2) || 1;
            const c1 = unigramCount.get(seg[i]) || 1;
            s2 = 3.2 + bc / c1;
          } else if (totalMorae > 20 && (bigramCount.get(w2) || 0) >= 2) {
            s2 = 1.1;
          }
          if (dp[i] + s2 > dp[i + 2]) {
            dp[i + 2] = dp[i] + s2;
            back[i + 2] = i;
          }
        }
      }

      const pieces = [];
      let pos = L;
      while (pos > 0) {
        const prev = back[pos];
        pieces.push(seg.slice(prev, pos).join(''));
        pos = prev;
      }
      pieces.reverse();

      const ids = [];
      for (const p of pieces) {
        if (wordToIdx.has(p)) ids.push(wordToIdx.get(p));
        else if (allowBirthUnknown && p.length <= 2) {
          const nid = birthNeuron(p);
          if (nid >= 0) ids.push(nid);
        }
      }
      if (ids.length > 0) clauseIdsList.push(ids);
    }
    return clauseIdsList;
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

  function sparsePropagate(srcVec, mat, dstVec, topK = 5) {
    const n = words.length;
    dstVec.fill(0);
    for (let pre = 0; pre < n; pre++) {
      const a = srcVec[pre];
      if (a <= 0.12) continue;
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

  function feelBody(bodyFeel = {}) {
    if (bodyFeel.samu !== undefined) kokoro[EMO_SAMU] = Math.max(0, Math.min(1, bodyFeel.samu));
    if (bodyFeel.hara !== undefined) kokoro[EMO_HARA] = Math.max(0, Math.min(1, bodyFeel.hara));
    if (bodyFeel.kowa !== undefined) kokoro[EMO_KOWA] = Math.max(0, Math.min(1, bodyFeel.kowa));
    if (bodyFeel.kini !== undefined) kokoro[EMO_KINI] = Math.max(0, Math.min(1, bodyFeel.kini));
    if (bodyFeel.yasu !== undefined) kokoro[EMO_YASU] = Math.max(0, Math.min(1, bodyFeel.yasu));
  }

  function getStage() {
    if (steps < 20) return 0;
    if (steps < 50) return 1;
    return 2;
  }

  /**
   * 語根 i と現在の「こころ」ベクトルの共鳴度 (cos類似度ライクな選択性スコア)
   */
  function kokoroResonance(wordIdx) {
    const base = wordIdx * NUM_EMO;
    let match = 0;
    let emoNorm = 0;
    for (let e = 0; e < NUM_EMO; e++) {
      const w = W_emo[base + e];
      match += kokoro[e] * w;
      emoNorm += w;
    }
    if (emoNorm < 0.05) return 0;
    return match / (emoNorm + 0.25);
  }

  function perceive50(inputIds, lr = 0.22, opts = {}) {
    const { isNewContext = false, allowBirthUnknown = false, bodyFeel = null } = opts;
    if (!isValid50(inputIds)) return { unknownIdx: -1 };

    if (bodyFeel) feelBody(bodyFeel);

    const morae = idsToMoraChars(inputIds);
    if (lr > 0) updatePhonemeStatsAndDiscover(morae);

    const clauses = segmentContinuous(morae, allowBirthUnknown);
    const allIds = clauses.flat();
    const n = words.length;

    if (!bodyFeel && allIds.length > 0) {
      for (let e = 0; e < NUM_EMO; e++) {
        let drive = 0;
        for (const id of allIds) drive += W_emo[id * NUM_EMO + e];
        if (drive > 0.1) {
          kokoro[e] = Math.min(1.0, kokoro[e] * 0.8 + (drive / allIds.length) * 0.35);
        }
      }
    }

    if (lr > 0 && clauses.length > 0) {
      const touchedAssoc = new Set();
      const touchedCausal = new Set();

      // W_emo: 強い「こころ」と同時発火した語根を結びつけ、他情動を競合抑圧
      for (const id of allIds) {
        const base = id * NUM_EMO;
        for (let e = 0; e < NUM_EMO; e++) {
          const kVal = kokoro[e];
          if (kVal >= 0.55) {
            W_emo[base + e] = Math.min(1.2, W_emo[base + e] + lr * 0.5 * kVal * (1.0 - W_emo[base + e]));
          } else {
            W_emo[base + e] = Math.max(0, W_emo[base + e] - lr * 0.12 * W_emo[base + e]);
          }
        }
      }

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

      if (!isNewContext && prevTurnIds.length > 0) {
        const curSet = new Set(allIds);
        for (const pre of prevTurnIds) {
          if (words[pre] === 'くる') continue;
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

      prevTurnIds = clauses[0].slice(0, 2);
      steps += 1;
    }

    // 脳内2手先ロールアウト（tanh飽和の後に「こころ共鳴」を加算して分岐をくっきり出す！）
    stateNow.fill(0);
    for (const id of allIds) stateNow[id] = 1.0;

    const tmpC1 = new Float32Array(MAX_NEURONS);
    const tmpA1 = new Float32Array(MAX_NEURONS);
    sparsePropagate(stateNow, W_causal, tmpC1, 6);
    sparsePropagate(stateNow, W_assoc, tmpA1, 6);

    for (let i = 0; i < n; i++) {
      const baseSignal = Math.tanh(tmpC1[i] * 0.95 + tmpA1[i] * 0.85);
      if (baseSignal > 0.05) {
        stateFuture1[i] = baseSignal + kokoroResonance(i) * 1.1;
      } else {
        stateFuture1[i] = 0;
      }
    }

    const tmpC2 = new Float32Array(MAX_NEURONS);
    const tmpA2 = new Float32Array(MAX_NEURONS);
    sparsePropagate(stateFuture1, W_causal, tmpC2, 5);
    sparsePropagate(stateFuture1, W_assoc, tmpA2, 5);
    for (let i = 0; i < n; i++) {
      const base2 = Math.tanh(tmpC2[i] * 0.95 + tmpA2[i] * 0.8);
      stateFuture2[i] = base2 > 0.05 ? base2 + kokoroResonance(i) * 0.85 : 0;
    }

    return { segmented: clauses.map((c) => c.map((i) => words[i])) };
  }

  function speak50() {
    const n = words.length;
    const stage = getStage();

    const topK = (vec, k, exclude = new Set()) => {
      const arr = [];
      for (let i = 0; i < n; i++) {
        if (i === IDX_NANI || exclude.has(i) || words[i] === 'くる') continue;
        if (vec[i] > 0.12) arr.push({ idx: i, val: vec[i] });
      }
      arr.sort((a, b) => b.val - a.val);
      return arr.slice(0, k);
    };

    const nowTop = topK(stateNow, 1);
    const f1Top = topK(stateFuture1, 2, new Set(nowTop.map((x) => x.idx)));
    const f2Top = topK(stateFuture2, 2, new Set([...nowTop.map((x) => x.idx), ...f1Top.map((x) => x.idx)]));

    const clauses = [];
    if (stage === 0) {
      const pick = f1Top[0] || nowTop[0];
      clauses.push(pick ? words[pick.idx] : 'なに');
    } else {
      if (f1Top.length >= 2) {
        clauses.push(`${words[f1Top[0].idx]}・${words[f1Top[1].idx]}`);
      } else if (nowTop[0] && f1Top[0]) {
        clauses.push(`${words[nowTop[0].idx]}・${words[f1Top[0].idx]}`);
      } else if (f1Top[0]) {
        clauses.push(words[f1Top[0].idx]);
      }

      if (stage >= 2 && f2Top.length >= 1) {
        const anchor = f1Top[0] || nowTop[0];
        if (f2Top.length >= 2 && f2Top[1].val > 0.25) {
          clauses.push(`${words[f2Top[0].idx]}・${words[f2Top[1].idx]}`);
        } else if (anchor && f2Top[0].val > 0.25) {
          clauses.push(`${words[anchor.idx]}・${words[f2Top[0].idx]}`);
        }
      }
    }

    if (clauses.length === 0) clauses.push('なに');
    return encode50(clauses.join('。') + '。', { appendEos: true });
  }

  function spontaneous50() {
    const n = words.length;
    let topEmo = 0;
    let topVal = -1;
    for (let e = 0; e < NUM_EMO; e++) {
      if (kokoro[e] > topVal) {
        topVal = kokoro[e];
        topEmo = e;
      }
    }

    if (topEmo === EMO_KINI && topVal > 0.6) {
      const targetWord = words[Math.max(1, words.length - 1)];
      return encode50(`${targetWord}・なに。`, { appendEos: true });
    }

    const scores = [];
    for (let i = 1; i < n; i++) {
      if (words[i] === 'くる') continue;
      const w = W_emo[i * NUM_EMO + topEmo];
      if (w > 0.08) scores.push({ idx: i, val: w * topVal });
    }
    scores.sort((a, b) => b.val - a.val);

    if (scores.length >= 2) {
      return encode50(`${words[scores[0].idx]}・${words[scores[1].idx]}。`, { appendEos: true });
    }
    if (scores.length === 1) {
      return encode50(`${words[scores[0].idx]}。`, { appendEos: true });
    }
    return encode50('なに。', { appendEos: true });
  }

  function probeWithKokoro(yamatoText, mood) {
    feelBody(mood);
    perceive50(encode50(yamatoText, { appendEos: true }), 0.0, { isNewContext: true });
    return decode50(speak50());
  }

  return {
    name,
    getStage,
    getSteps: () => steps,
    getDiscoveredWords: () => words.slice(1),
    getKokoro: () => Object.fromEntries(KOKORO_NAMES.map((k, i) => [k, Number(kokoro[i].toFixed(2))])),
    feelBody,
    perceive50,
    speak50,
    spontaneous50,
    probeWithKokoro,
  };
}
