// chigo_v3.mjs — 非LLM・自己組織化 ニューラルちご (v3)
// v2の知見（「長く聞かせると全結合して飽和崩壊する」）を克服するための3大改良:
//   1. LTP + LTD（長期増強＋長期抑圧）: 発火した pre に対し、同時に発火しなかった post への結合を弱化
//   2. 恒常性スケーリング＆シナプス刈り込み: 各ニューロンの出力シナプス総量を一定に保ち、微小結合を0へ刈り込む
//   3. 側抑制（スパース活性化伝播）: 伝播ステップごとに上位ニューロン以外を抑制し、脳全体のてんかん発火を防ぐ
// 入出力は codec50.mjs の50音ID列 (0..49) のみ、語彙は1〜2文字語に限定。

import { encode50, decode50, isValid50 } from './codec50.mjs';
import { createRng } from './jomon_world.mjs';
import { NEURON_WORDS, NUM_NEURONS } from './neural_baby_v2.mjs';

const WORD_TO_IDX = new Map(NEURON_WORDS.map((w, i) => [w, i]));
const THING_END = 14;  // 0..13: もの/自然
const PRED_START = 14; // 14..31: 感覚/動き
const IDX_NANI = WORD_TO_IDX.get('なに');

/**
 * 50音デコード文字列を「句（。区切り）」ごとに分解し、各句のニューロンID配列を返す。
 */
function parseClausesToIndices(text) {
  const rawClauses = text.split('。').map((s) => s.trim()).filter(Boolean);
  const clauses = [];

  for (const clause of rawClauses) {
    const chunks = clause.split('・').map((s) => s.trim()).filter(Boolean);
    const ids = [];
    for (const ch of chunks) {
      if (WORD_TO_IDX.has(ch)) {
        ids.push(WORD_TO_IDX.get(ch));
        continue;
      }
      let i = 0;
      while (i < ch.length) {
        const two = ch.slice(i, i + 2);
        const one = ch.slice(i, i + 1);
        if (two.length === 2 && WORD_TO_IDX.has(two)) {
          ids.push(WORD_TO_IDX.get(two));
          i += 2;
        } else if (WORD_TO_IDX.has(one)) {
          ids.push(WORD_TO_IDX.get(one));
          i += 1;
        } else {
          i += 1;
        }
      }
    }
    if (ids.length > 0) clauses.push(ids);
  }
  return clauses;
}

export function createChigoV3(opts = {}) {
  const { name = 'ちごv3', seed = 101 } = opts;
  const rng = createRng(seed);

  // シナプス結合行列 W[pre * N + post]（真っ白な微小ノイズから開始）
  const W = new Float32Array(NUM_NEURONS * NUM_NEURONS);
  for (let i = 0; i < W.length; i++) {
    W[i] = rng() * 0.01;
  }

  const act = new Float32Array(NUM_NEURONS);
  const pred = new Float32Array(NUM_NEURONS);
  const fatigue = new Float32Array(NUM_NEURONS);

  let steps = 0;
  let prevTopicIds = [];

  function getStage() {
    if (steps < 12) return 0; // 赤ん坊: 一語文
    if (steps < 28) return 1; // 幼児: 二語文
    return 2; // 発達ちご: 二語文＋連鎖
  }

  /**
   * 恒常性スケーリング & シナプス刈り込み（1ニューロンあたりの結合予算を一定に保つ）
   */
  function normalizeAndPruneRow(pre) {
    const row = pre * NUM_NEURONS;
    let sum = 0;
    for (let j = 0; j < NUM_NEURONS; j++) {
      let v = W[row + j];
      if (v < 0.015) v = 0; // 微小シナプスを刈り込む
      W[row + j] = v;
      sum += v;
    }
    // 出力シナプス総量の上限 (budget = 2.2) を超えたら全体を比例スケーリング
    const BUDGET = 2.2;
    if (sum > BUDGET) {
      const scale = BUDGET / sum;
      for (let j = 0; j < NUM_NEURONS; j++) {
        W[row + j] *= scale;
      }
    }
  }

  /**
   * 側抑制（上位K個のみを残して他を抑え込むスパース化）
   */
  function lateralInhibition(vec, topK = 5) {
    const vals = Array.from(vec);
    vals.sort((a, b) => b - a);
    const thresh = vals[Math.min(topK, vals.length - 1)] || 0;
    for (let i = 0; i < vec.length; i++) {
      if (vec[i] < thresh) {
        vec[i] *= 0.15;
      }
    }
  }

  /**
   * 50音ID列を聴取し、LTP/LTD・刈り込み・スパース活性化伝播を行う。
   */
  function perceive50(inputIds, lr = 0.2, opts = {}) {
    const { resetContext = false } = opts;
    if (!isValid50(inputIds)) return { surprise: 0, heardIndices: [] };
    if (resetContext) {
      act.fill(0);
      fatigue.fill(0);
    }

    const text = decode50(inputIds);
    const clauses = parseClausesToIndices(text);
    const allHeard = clauses.flat();
    const heardSet = new Set(allHeard);

    // 1) 予測誤差（驚き）の計算
    pred.fill(0);
    for (let pre = 0; pre < NUM_NEURONS; pre++) {
      const a = act[pre];
      if (a <= 0.1) continue;
      const row = pre * NUM_NEURONS;
      for (let post = 0; post < NUM_NEURONS; post++) {
        pred[post] += a * W[row + post];
      }
    }
    let surprise = 0;
    for (const idx of allHeard) {
      surprise += Math.max(0, 1.0 - Math.max(0, pred[idx]));
    }
    surprise = allHeard.length > 0 ? surprise / allHeard.length : 0;

    // 2) シナプス更新 (lr > 0 のときだけ学習)
    if (lr > 0 && clauses.length > 0) {
      const touchedPres = new Set();

      // 2a) 句内（例: 「くま・こわ」「しか・とる」）の強い有向ヘッブ学習 (LTP) + 競合抑圧 (LTD)
      for (const clause of clauses) {
        const clauseSet = new Set(clause);
        for (let i = 0; i < clause.length; i++) {
          const pre = clause[i];
          const row = pre * NUM_NEURONS;
          touchedPres.add(pre);

          // LTD: pre が発火したのに同じ句に現れなかった相手への結合を少し弱める
          for (let j = 0; j < NUM_NEURONS; j++) {
            if (!clauseSet.has(j) && W[row + j] > 0) {
              W[row + j] = Math.max(0, W[row + j] - lr * 0.12 * W[row + j]);
            }
          }

          // LTP: 句内で共起した相手（特に後ろに来る語）を強く結ぶ
          for (let j = 0; j < clause.length; j++) {
            if (i === j) continue;
            const post = clause[j];
            const dir = i < j ? 1.0 : 0.35;
            W[row + post] = Math.min(1.5, W[row + post] + lr * dir * (1.2 - W[row + post]));
          }
        }
      }

      // 2b) 同一発話内の2句間（例: 「くま・こわ。ひと・にげ。」→ くま→にげ）と
      //     直前トピック（例: 「あめ・くる」→「ひ・きえ」）の弱めな因果結合
      if (clauses.length >= 2) {
        const firstSubj = clauses[0][0];
        const secondPred = clauses[1][clauses[1].length - 1];
        if (firstSubj !== secondPred) {
          const idx = firstSubj * NUM_NEURONS + secondPred;
          W[idx] = Math.min(1.5, W[idx] + lr * 0.65 * (1.2 - W[idx]));
          touchedPres.add(firstSubj);
        }
      }
      for (const prevId of prevTopicIds) {
        const row = prevId * NUM_NEURONS;
        touchedPres.add(prevId);
        for (const post of allHeard) {
          if (prevId === post) continue;
          W[row + post] = Math.min(1.5, W[row + post] + lr * 0.25 * (1.0 - W[row + post]));
        }
      }

      // 2c) 恒常性スケーリング＆刈り込み
      for (const pre of touchedPres) {
        normalizeAndPruneRow(pre);
      }

      prevTopicIds = clauses[0].slice(0, 2);
      steps += 1;
    }

    // 3) 活性化更新（直前残響を抑えて入力をくっきり反映）
    for (let i = 0; i < NUM_NEURONS; i++) {
      act[i] = resetContext ? 0 : act[i] * 0.18;
      fatigue[i] *= 0.6;
    }
    for (const idx of heardSet) {
      act[idx] = 1.0;
    }

    // 4) スパース活性化伝播（側抑制つき 2ホップ）
    const nextAct = new Float32Array(NUM_NEURONS);
    for (let hop = 0; hop < 2; hop++) {
      nextAct.set(act);
      for (let pre = 0; pre < NUM_NEURONS; pre++) {
        const a = act[pre];
        if (a <= 0.15) continue;
        const row = pre * NUM_NEURONS;
        for (let post = 0; post < NUM_NEURONS; post++) {
          if (pre === post) continue;
          const w = W[row + post];
          if (w > 0.02) {
            nextAct[post] += a * w * (hop === 0 ? 0.65 : 0.22);
          }
        }
      }
      lateralInhibition(nextAct, 6);
      for (let i = 0; i < NUM_NEURONS; i++) {
        act[i] = Math.tanh(nextAct[i]);
      }
    }

    // 未知語（シナプス総量がほぼ0）を聞いたら「なに」が発火
    if (allHeard.length > 0) {
      let maxKnown = 0;
      for (const idx of allHeard) {
        const row = idx * NUM_NEURONS;
        let rowSum = 0;
        for (let j = 0; j < NUM_NEURONS; j++) rowSum += W[row + j];
        if (rowSum > maxKnown) maxKnown = rowSum;
      }
      if (maxKnown < 0.2) {
        act[IDX_NANI] = Math.max(act[IDX_NANI], 0.9);
      }
    }

    return { surprise, heardIndices: allHeard };
  }

  /**
   * 現在のスパース活性から50音ID列を発話する（1〜2文字語のみ）。
   */
  function speak50() {
    const stage = getStage();
    const score = new Float32Array(NUM_NEURONS);
    for (let i = 0; i < NUM_NEURONS; i++) {
      score[i] = act[i] - fatigue[i] * 0.45;
    }

    const pickTop = (start, end, exclude = -1) => {
      let bestIdx = -1;
      let bestVal = -Infinity;
      for (let i = start; i < end; i++) {
        if (i === exclude) continue;
        if (score[i] > bestVal) {
          bestVal = score[i];
          bestIdx = i;
        }
      }
      return { idx: bestIdx, val: bestVal };
    };

    const spoken = [];
    if (stage === 0) {
      const top = pickTop(0, NUM_NEURONS);
      if (top.idx >= 0) {
        spoken.push(NEURON_WORDS[top.idx]);
        fatigue[top.idx] += 0.6;
      }
    } else if (stage === 1) {
      const thing = pickTop(0, THING_END);
      const predTok = pickTop(PRED_START, NUM_NEURONS);
      if (thing.val > 0.2 && predTok.val > 0.15) {
        spoken.push(`${NEURON_WORDS[thing.idx]}・${NEURON_WORDS[predTok.idx]}`);
      } else {
        const top = pickTop(0, NUM_NEURONS);
        spoken.push(NEURON_WORDS[top.idx]);
      }
    } else {
      const thing = pickTop(0, THING_END);
      const pred1 = pickTop(PRED_START, NUM_NEURONS);
      const pred2 = pickTop(PRED_START, NUM_NEURONS, pred1.idx);

      if (thing.val > 0.2 && pred1.val > 0.15) {
        spoken.push(`${NEURON_WORDS[thing.idx]}・${NEURON_WORDS[pred1.idx]}`);
        if (pred2.val > 0.38) {
          const thing2 = pickTop(0, THING_END, thing.idx);
          const subj2 = thing2.val > 0.35 ? NEURON_WORDS[thing2.idx] : NEURON_WORDS[thing.idx];
          spoken.push(`${subj2}・${NEURON_WORDS[pred2.idx]}`);
        }
      } else {
        const top = pickTop(0, NUM_NEURONS);
        spoken.push(NEURON_WORDS[top.idx]);
      }
    }

    return encode50(spoken.join('。') + '。', { appendEos: true });
  }

  /**
   * 学習を止め、残響をクリアして純粋な応答をテストする。
   */
  function probeText(yamatoText) {
    const inIds = encode50(yamatoText, { appendEos: true });
    perceive50(inIds, 0.0, { resetContext: true });
    return decode50(speak50());
  }

  function getWeight(wordA, wordB) {
    const a = WORD_TO_IDX.get(wordA);
    const b = WORD_TO_IDX.get(wordB);
    if (a === undefined || b === undefined) return 0;
    return Number(W[a * NUM_NEURONS + b].toFixed(3));
  }

  /**
   * シナプスの疎さ（0でない結合の割合）を返す。飽和監視用。
   */
  function getDensity() {
    let active = 0;
    for (let i = 0; i < W.length; i++) {
      if (W[i] > 0.05) active += 1;
    }
    return Number(((active / W.length) * 100).toFixed(1));
  }

  return {
    name,
    getStage,
    getSteps: () => steps,
    getWeight,
    getDensity,
    perceive50,
    speak50,
    probeText,
  };
}
