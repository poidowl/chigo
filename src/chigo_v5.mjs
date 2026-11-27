// chigo_v5.mjs — 非LLM・初期辞書ゼロ・連続50音からの語根自己発見＋2系統STDP 原始AGI「ちご v5」
// v4からの大進化:
//   1. 初期辞書ゼロ（NEURON_WORDS を使わない！初期語彙は「なに」のみ）
//   2. 入力は区切り記号(・)を一切含まない「生の連続50音ID列 (0..49)」をそのまま受け取る
//   3. 赤ん坊の統計的言語獲得（Saffran 遷移確率 & PMI: Float32Array(50*50)）で、
//      繰り返し一緒に鳴る2音節の塊（あめ・くま・しか・しま・くに等）と高頻度1音節（ひ・き・み・よ）を
//      自力で切り出し、概念ニューロンとして動的に誕生させる（3文字以上は封印）
//   4. 誕生したニューロン同士を v4 の2系統シナプス (W_assoc / W_causal) で結び、
//      ターン跨ぎの因果予測＋2手先ロールアウト＋未知語ワンショット学習を行う。

import { encode50, decode50, isValid50, VOCAB_SIZE, TOKEN_DAKUTEN, TOKEN_SEP, TOKEN_EOS, TOKEN_PAD } from './codec50.mjs';

const MAX_NEURONS = 96;

/**
 * 50音ID列 (0..49) を「1モーラ単位の文字列配列」に変換する。
 * 例: [か, ゛, わ] -> ['が', 'わ']（濁点は直前の清音に結合して1文字＝1モーラとして扱う）
 */
function idsToMoraChars(ids) {
  const text = decode50(ids);
  const morae = [];
  for (const ch of text) {
    if (ch === '_' || ch === '・' || ch === '。') {
      morae.push('。'); // 発話区切りマーカー
    } else {
      morae.push(ch);
    }
  }
  return morae;
}

export function createChigoV5(opts = {}) {
  const { name = 'ちごv5' } = opts;

  // 音韻統計（生の連続50音から語根を発見するための1文字・2文字カウント）
  const unigramCount = new Map();  // char -> count
  const bigramCount = new Map();   // "c1c2" -> count
  let totalMorae = 0;

  // 動的ニューロン辞書（初期は「なに」のみ！他はすべて聴取から自力発見）
  const words = ['なに'];
  const wordToIdx = new Map([['なに', 0]]);
  const wordFreq = new Float32Array(MAX_NEURONS);
  const IDX_NANI = 0;

  // 2系統シナプス (MAX_NEURONS x MAX_NEURONS)
  const W_assoc = new Float32Array(MAX_NEURONS * MAX_NEURONS);
  const W_causal = new Float32Array(MAX_NEURONS * MAX_NEURONS);

  // 脳内状態ベクトル
  const stateNow = new Float32Array(MAX_NEURONS);
  const stateFuture1 = new Float32Array(MAX_NEURONS);
  const stateFuture2 = new Float32Array(MAX_NEURONS);

  let steps = 0;
  let prevTurnIds = [];
  let pendingUnknownIdx = -1;

  function birthNeuron(word) {
    if (!word || word.length < 1 || word.length > 2) return -1;
    if (wordToIdx.has(word)) return wordToIdx.get(word);
    if (words.length >= MAX_NEURONS) return -1;
    const idx = words.length;
    words.push(word);
    wordToIdx.set(word, idx);
    return idx;
  }

  /**
   * 連続モーラ列の統計を更新し、結合度（PMI・遷移確率）が高い2文字語根と
   * 独立出現率の高い1文字語根を自動でニューロン化する（教師なし分節）。
   */
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

        // 2文字語根の自己発見条件:
        // 複数回出現し、かつ P(c2|c1) または P(c1|c2) の結合度（遷移確率）が十分高い
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

      // 1文字語根の自己発見:
      // 句頭や単独で何度も現れ、特定の後続文字に固定されていない音（ひ・き・み・よ 等）
      const prevBoundary = i === 0 || morae[i - 1] === '。';
      const nextBoundary = i + 1 >= morae.length || morae[i + 1] === '。';
      if ((prevBoundary && nextBoundary) || (unigramCount.get(c1) >= 8 && prevBoundary)) {
        // 強い2文字語根の先頭ばかりでない場合のみ1文字語として誕生
        if (!wordToIdx.has(c1)) {
          let maxFwd = 0;
          for (const [bi, bc] of bigramCount.entries()) {
            if (bi[0] === c1) {
              const tp = bc / (unigramCount.get(c1) || 1);
              if (tp > maxFwd) maxFwd = tp;
            }
          }
          if (maxFwd < 0.55) {
            birthNeuron(c1);
          }
        }
      }
    }
  }

  /**
   * 連続モーラ列を「発見済み語根（最大2文字）＋遷移確率の谷」で自力分節する。
   * 区切り記号（・）が全く無くても、統計的結合度スコアが最大の切り方（動的計画法）を選ぶ！
   */
  function segmentContinuous(morae, allowBirthUnknown = false) {
    // 発話区切り '。' ごとに分割
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
      // 1〜2文字の最良分節を動的計画法 (Viterbi-like) で求める
      const dp = new Float64Array(L + 1).fill(-1e9);
      const back = new Int32Array(L + 1).fill(-1);
      dp[0] = 0;

      for (let i = 0; i < L; i++) {
        if (dp[i] < -1e8) continue;

        // 1文字候補
        const w1 = seg[i];
        const score1 = wordToIdx.has(w1) ? 1.6 : 0.1;
        if (dp[i] + score1 > dp[i + 1]) {
          dp[i + 1] = dp[i] + score1;
          back[i + 1] = i;
        }

        // 2文字候補
        if (i + 2 <= L) {
          const w2 = seg[i] + seg[i + 1];
          let score2 = 0.0;
          if (wordToIdx.has(w2)) {
            const bc = bigramCount.get(w2) || 1;
            const c1 = unigramCount.get(seg[i]) || 1;
            const tp = bc / c1;
            score2 = 3.2 + tp; // 発見済み2文字語根を優先
          } else if (totalMorae > 20) {
            const bc = bigramCount.get(w2) || 0;
            if (bc >= 2) score2 = 1.1;
          }
          if (dp[i] + score2 > dp[i + 2]) {
            dp[i + 2] = dp[i] + score2;
            back[i + 2] = i;
          }
        }
      }

      // バックトラックして切り出し結果を得る
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
        if (wordToIdx.has(p)) {
          ids.push(wordToIdx.get(p));
        } else if (allowBirthUnknown && p.length <= 2) {
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
    if (steps < 20) return 0; // 赤ん坊: 音の塊を発見中・一語文
    if (steps < 50) return 1; // 幼児: 二語文
    return 2; // 原始AGI: 連続音から発見した語根で2手先推論
  }

  /**
   * 生の連続50音ID列 (0..49) を聴取し、
   * 1) 統計的語根発見 -> 2) 自力分節 -> 3) 2系統STDP学習 -> 4) 2手先ロールアウト
   */
  function perceive50(inputIds, lr = 0.22, opts = {}) {
    const { isNewContext = false, allowBirthUnknown = false } = opts;
    if (!isValid50(inputIds)) return { unknownIdx: -1 };

    const morae = idsToMoraChars(inputIds);
    if (lr > 0) {
      updatePhonemeStatsAndDiscover(morae);
    }

    const clauses = segmentContinuous(morae, allowBirthUnknown);
    const allIds = clauses.flat();
    const n = words.length;

    // 未知語への教えを受け取った場合 → ワンショット刻印
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

    // 未知語検出（対話時）
    let detectedUnknown = -1;
    if (allowBirthUnknown && learnedUnknownNow < 0) {
      for (const id of allIds) {
        if (id !== IDX_NANI && rowSynapseStrength(id) < 0.05) {
          detectedUnknown = id;
          break;
        }
      }
    }

    // 2系統STDP学習
    if (lr > 0 && clauses.length > 0 && detectedUnknown < 0 && learnedUnknownNow < 0) {
      const touchedAssoc = new Set();
      const touchedCausal = new Set();

      for (const id of allIds) wordFreq[id] += 1;

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

    // 脳内2手先ロールアウト
    stateNow.fill(0);
    if (learnedUnknownNow >= 0) {
      stateNow[learnedUnknownNow] = 1.0;
    } else {
      for (const id of allIds) stateNow[id] = 1.0;
    }

    const tmpC1 = new Float32Array(MAX_NEURONS);
    const tmpA1 = new Float32Array(MAX_NEURONS);
    sparsePropagate(stateNow, W_causal, tmpC1, 3);
    sparsePropagate(stateNow, W_assoc, tmpA1, 3);
    for (let i = 0; i < n; i++) {
      stateFuture1[i] = Math.tanh(tmpC1[i] * 1.0 + tmpA1[i] * 0.85);
    }

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

    return { unknownIdx: detectedUnknown, segmented: clauses.map((c) => c.map((i) => words[i])) };
  }

  function speak50() {
    const n = words.length;
    const stage = getStage();

    if (pendingUnknownIdx >= 0) {
      const unk = words[pendingUnknownIdx];
      return encode50(stage === 0 ? `${unk}。なに。` : `${unk}・なに。`, { appendEos: true });
    }

    // 語種辞書（品詞）すら持たないので、純粋に活性の強さ順でペアを組む！
    const topK = (vec, k, exclude = new Set()) => {
      const arr = [];
      for (let i = 0; i < n; i++) {
        if (i === IDX_NANI || exclude.has(i) || words[i] === 'くる') continue;
        if (vec[i] > 0.15) arr.push({ idx: i, val: vec[i] });
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

  function probeText(yamatoText) {
    pendingUnknownIdx = -1;
    const inIds = encode50(yamatoText, { appendEos: true });
    perceive50(inIds, 0.0, { isNewContext: true, allowBirthUnknown: false });
    return decode50(speak50());
  }

  function talkText(yamatoText) {
    const inIds = encode50(yamatoText, { appendEos: true });
    perceive50(inIds, 0.25, { isNewContext: true, allowBirthUnknown: true });
    return decode50(speak50());
  }

  return {
    name,
    getStage,
    getSteps: () => steps,
    getDiscoveredWords: () => words.slice(1), // 初期の「なに」を除いた自力発見語彙
    segmentText: (text) => {
      const morae = idsToMoraChars(encode50(text, { appendEos: true }));
      return segmentContinuous(morae, false).map((c) => c.map((i) => words[i]));
    },
    perceive50,
    speak50,
    probeText,
    talkText,
  };
}
