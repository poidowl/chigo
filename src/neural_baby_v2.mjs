// neural_baby_v2.mjs — 非LLM・自己組織化 ニューラルベビー (v2)
// 仕組み:
//   - Float32Array の再帰的結合行列 W (N x N) による活性化伝播 (Spreading Activation)
//   - ヘッブ学習 (Hebbian + 減衰正規化) で体験・会話の共起からシナプスが勝手に育つ
//   - 予測誤差（驚き）が大きいと「なに」ニューロンが発火して好奇心を示す
//   - 神経疲労（不応期 fatigue）により同じ単語のループを防ぎ、連想が自然に流れる
//   - 入出力は codec50.mjs の50音ID列 (0..49) のみ、語彙は1〜2文字語に限定

import { encode50, decode50, isValid50 } from './codec50.mjs';
import { createRng } from './jomon_world.mjs';

// ニューロンに対応する1〜2文字のヤマトコトバ（3文字以上は封印）
export const NEURON_WORDS = [
  // 自然・もの (0..13)
  'ひ', 'き', 'み', 'みず', 'あめ', 'かぜ', 'やま', 'かわ',
  'くま', 'しか', 'うお', 'ひと', 'よ', 'ほし',
  // 感覚・快不快 (14..21)
  'あつ', 'さむ', 'こわ', 'うま', 'いた', 'よい', 'やだ', 'はら',
  // 動き・変化 (22..31)
  'もえ', 'きえ', 'くる', 'ゆく', 'くう', 'のむ', 'ねる', 'にげ',
  'たす', 'とる',
  // 問い (32..33)
  'なに', 'なぜ',
];

export const NUM_NEURONS = NEURON_WORDS.length;
const WORD_TO_IDX = new Map(NEURON_WORDS.map((w, i) => [w, i]));

const THING_END = 14; // 0..13: もの/自然
const PRED_START = 14; // 14..31: 感覚/動き
const IDX_NANI = WORD_TO_IDX.get('なに');

function extractNeuronIndices(text) {
  const chunks = text
    .replace(/。/g, '・')
    .split('・')
    .map((s) => s.trim())
    .filter(Boolean);

  const hits = [];
  for (const ch of chunks) {
    if (WORD_TO_IDX.has(ch)) {
      hits.push(WORD_TO_IDX.get(ch));
      continue;
    }
    // 連結されている場合は最長2文字マッチで切り出す
    let i = 0;
    while (i < ch.length) {
      const two = ch.slice(i, i + 2);
      const one = ch.slice(i, i + 1);
      if (two.length === 2 && WORD_TO_IDX.has(two)) {
        hits.push(WORD_TO_IDX.get(two));
        i += 2;
      } else if (WORD_TO_IDX.has(one)) {
        hits.push(WORD_TO_IDX.get(one));
        i += 1;
      } else {
        i += 1;
      }
    }
  }
  return hits;
}

export function createNeuralBabyV2(opts = {}) {
  const { name = 'べびー', seed = 1 } = opts;
  const rng = createRng(seed);

  // シナプス結合行列 W[pre * N + post] (初期値はごく小さなランダムノイズのみ＝知識ほぼゼロ！)
  const W = new Float32Array(NUM_NEURONS * NUM_NEURONS);
  for (let i = 0; i < W.length; i++) {
    W[i] = (rng() - 0.5) * 0.04;
  }

  // 現在の神経活性・予測活性・不応期疲労
  const act = new Float32Array(NUM_NEURONS);
  const pred = new Float32Array(NUM_NEURONS);
  const fatigue = new Float32Array(NUM_NEURONS);

  let steps = 0;
  let totalSurprise = 0;
  let totalSynapseGrowth = 0;

  function getStage() {
    // 体験ステップとシナプス成熟度で赤ん坊→幼児→原始人へ発達
    if (steps < 12) return 0; // 赤ん坊: 一語文
    if (steps < 28) return 1; // 幼児: 二語文の芽生え
    return 2; // 原始人ベビー: 二語文の掛け合い・連鎖
  }

  /**
   * 外部刺激（環境体験や相手の発話）を50音ID列として受け取り、
   * 予測誤差を計算し、ヘッブ学習でシナプスを更新する。
   */
  function perceive50(inputIds, lr = 0.22) {
    if (!isValid50(inputIds)) return { surprise: 0, heardIndices: [] };
    const text = decode50(inputIds);
    const heard = extractNeuronIndices(text);

    // 1) 直前の活性化状態から「次に何が来るか」の予測ベクトルを作る
    pred.fill(0);
    for (let pre = 0; pre < NUM_NEURONS; pre++) {
      const a = act[pre];
      if (a <= 0.05) continue;
      const row = pre * NUM_NEURONS;
      for (let post = 0; post < NUM_NEURONS; post++) {
        pred[post] += a * W[row + post];
      }
    }

    // 2) 実入力のベクトル化 & 予測誤差（驚き）計算
    const target = new Float32Array(NUM_NEURONS);
    for (const idx of heard) target[idx] = 1.0;

    let surprise = 0;
    for (const idx of heard) {
      const err = Math.max(0, 1.0 - Math.max(0, pred[idx]));
      surprise += err;
    }
    surprise = heard.length > 0 ? surprise / heard.length : 0;
    totalSurprise += surprise;

    // 3) 時間差ヘッブ学習 (直前の活性 pre -> 今回の入力 post)
    //    さらに今回同時に聞いた語同士 (co-occurrence) も結合する
    for (let pre = 0; pre < NUM_NEURONS; pre++) {
      const aPre = act[pre];
      if (aPre <= 0.08) continue;
      const row = pre * NUM_NEURONS;
      for (const post of heard) {
        if (pre === post) continue;
        const delta = lr * aPre * (1.0 - W[row + post]);
        W[row + post] = Math.min(1.2, W[row + post] + delta);
        totalSynapseGrowth += Math.abs(delta);
      }
    }

    for (let i = 0; i < heard.length; i++) {
      for (let j = 0; j < heard.length; j++) {
        if (i === j) continue;
        const u = heard[i];
        const v = heard[j];
        // 語順の向き（u -> v）を少し強めに刻む（因果・二語文の芽）
        const dirBoost = i < j ? 1.0 : 0.45;
        const idx = u * NUM_NEURONS + v;
        const delta = lr * dirBoost * (1.0 - W[idx]);
        W[idx] = Math.min(1.2, W[idx] + delta);
        totalSynapseGrowth += Math.abs(delta);
      }
    }

    // 4) ニューロン活性を更新（漏れ積分 + 新規入力 + わずかな自然減衰）
    for (let i = 0; i < NUM_NEURONS; i++) {
      act[i] = act[i] * 0.35 + target[i] * 0.95;
      fatigue[i] *= 0.65; // 不応期の回復
    }

    // 5) ネットワーク内で活性化伝播 (2ホップの連想波)
    const nextAct = new Float32Array(NUM_NEURONS);
    for (let hop = 0; hop < 2; hop++) {
      nextAct.set(act);
      for (let pre = 0; pre < NUM_NEURONS; pre++) {
        const a = act[pre];
        if (a <= 0.1) continue;
        const row = pre * NUM_NEURONS;
        for (let post = 0; post < NUM_NEURONS; post++) {
          if (pre === post) continue;
          nextAct[post] += a * W[row + post] * (hop === 0 ? 0.55 : 0.25);
        }
      }
      for (let i = 0; i < NUM_NEURONS; i++) {
        act[i] = Math.tanh(nextAct[i]);
      }
    }

    // 驚きが大きいときは好奇心ニューロン「なに」が発火
    if (surprise > 0.78 && heard.length > 0) {
      act[IDX_NANI] = Math.max(act[IDX_NANI], surprise * 0.85);
    }

    steps += 1;
    return { surprise, heardIndices: heard };
  }

  /**
   * 現在の脳内活性から50音ID列を発話する（3文字以上は物理的に出力不可）。
   */
  function speak50() {
    const stage = getStage();

    // 不応期（最近喋った語）を引いた実効スコアで発話ニューロンを選ぶ
    const score = new Float32Array(NUM_NEURONS);
    for (let i = 0; i < NUM_NEURONS; i++) {
      score[i] = act[i] - fatigue[i] * 0.55 + (rng() - 0.5) * 0.03;
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

    let spokenWords = [];

    if (stage === 0) {
      // Stage 0（赤ん坊）: 一番強く発火した1語だけ（一語文）
      const top = pickTop(0, NUM_NEURONS);
      if (top.idx >= 0) {
        spokenWords = [NEURON_WORDS[top.idx]];
        fatigue[top.idx] = Math.min(1.5, fatigue[top.idx] + 0.8);
      }
    } else if (stage === 1) {
      // Stage 1（幼児）: 最も強い「もの」+ 最も強い「感覚/動き/問い」の二語文
      const thing = pickTop(0, THING_END);
      const predTok = pickTop(PRED_START, NUM_NEURONS);
      if (thing.val > 0.2 && predTok.val > 0.15) {
        spokenWords = [`${NEURON_WORDS[thing.idx]}・${NEURON_WORDS[predTok.idx]}`];
        fatigue[thing.idx] += 0.75;
        fatigue[predTok.idx] += 0.75;
      } else {
        const top = pickTop(0, NUM_NEURONS);
        spokenWords = [NEURON_WORDS[top.idx]];
        fatigue[top.idx] += 0.75;
      }
    } else {
      // Stage 2（原始人ベビー）: 対象＋予測/行動の二語文。
      // さらに強く連鎖発火した2つ目の行動・感覚があれば2句目も繋げる。
      const thing = pickTop(0, THING_END);
      const pred1 = pickTop(PRED_START, NUM_NEURONS);
      const pred2 = pickTop(PRED_START, NUM_NEURONS, pred1.idx);

      if (thing.val > 0.15 && pred1.val > 0.15) {
        spokenWords.push(`${NEURON_WORDS[thing.idx]}・${NEURON_WORDS[pred1.idx]}`);
        fatigue[thing.idx] += 0.8;
        fatigue[pred1.idx] += 0.8;

        // 2つ目の強い連想があれば二語文を追加（ただし連発しすぎない閾値）
        if (pred2.val > 0.48) {
          const thing2 = pickTop(0, THING_END, thing.idx);
          const subj2 = thing2.val > 0.35 ? NEURON_WORDS[thing2.idx] : NEURON_WORDS[thing.idx];
          spokenWords.push(`${subj2}・${NEURON_WORDS[pred2.idx]}`);
          fatigue[pred2.idx] += 0.8;
        }
      } else {
        const top = pickTop(0, NUM_NEURONS);
        spokenWords.push(NEURON_WORDS[top.idx]);
        fatigue[top.idx] += 0.8;
      }
    }

    const utterance = spokenWords.join('。') + '。';
    return encode50(utterance, { appendEos: true });
  }

  /**
   * 特定の2語間のシナプス結合強度を観察用に取り出す。
   */
  function getWeight(wordA, wordB) {
    const a = WORD_TO_IDX.get(wordA);
    const b = WORD_TO_IDX.get(wordB);
    if (a === undefined || b === undefined) return 0;
    return Number(W[a * NUM_NEURONS + b].toFixed(3));
  }

  return {
    name,
    getStage,
    getSteps: () => steps,
    getAvgSurprise: () => (steps > 0 ? totalSurprise / steps : 0),
    getWeight,
    perceive50,
    speak50,
    // 便利メソッド（人間・環境からの体験注入）
    experienceText(yamatoText, lr = 0.28) {
      return perceive50(encode50(yamatoText, { appendEos: true }), lr);
    },
    listenAndReply50(inputIds) {
      const { surprise } = perceive50(inputIds, 0.2);
      const outIds = speak50();
      return {
        outIds,
        reply: decode50(outIds),
        stage: getStage(),
        surprise: Number(surprise.toFixed(2)),
      };
    },
  };
}