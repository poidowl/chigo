// chigo_v16.mjs — 純ニューラルちご v16「人間語アライメント（言語ドリフト対策）＆バイリンガル通訳脳」
// 100%純ニューラル・ゼロ依存・厳密50音I/O (0..49)
//
// v15からの大進化（宇宙人語へのドリフトを防ぎ、人間と話が通じる創発社会へ）:
//   1. 長老アンカー方式 (Human-in-the-Loop Grounding):
//      ちご同士の協力ゲームの中に、人間の長老（2文字ヤマトコトバを話す固定エージェント）を
//      一定割合 (25%) 混ぜるだけで、創発語彙が宇宙人語へドリフトせず人間のかなへ自動収束！
//      しかも人間が教えていない未知事象（7番目の危機）だけはちご同士が自力で新語を発明する。
//   2. バイリンガル通訳脳 (Pure-Neural 50-kana Translator):
//      すでに独自語（わし。/つろ。等）へドリフトしてしまった v15 ちごに対しても、
//      共通体験から「ちご語 ⇔ 人間語」を双方向に翻訳する純ニューラル通訳器を獲得できる。

import { VOCAB_SIZE, TOKEN_EOS, encode50, decode50 } from './codec50.mjs';
import { createRng } from './jomon_world.mjs';

const KANA_START = 4;
const KANA_COUNT = VOCAB_SIZE - KANA_START; // 46

function softmaxArr(logits, out) {
  let maxL = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > maxL) maxL = logits[i];
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - maxL);
    out[i] = e;
    sum += e;
  }
  const inv = 1 / sum;
  for (let i = 0; i < logits.length; i++) out[i] *= inv;
}

function argmaxArr(arr) {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

/**
 * 村の人間（長老）エージェント:
 * 既知の人間かな語彙（例: くま。/しか。/ひを。/あめ。/うお。/みを。）だけを話し、理解する。
 * 未知の宇宙人語を聞いた場合は理解できず -1（誤解）を返す。
 */
export function createHumanElderV16(humanWords) {
  const encoded = humanWords.map((w) => encode50(w, { appendEos: true }));

  function speak(idx) {
    if (idx < 0 || idx >= encoded.length) return null;
    return encoded[idx];
  }

  function understand(msgIds) {
    const text = decode50(msgIds);
    for (let i = 0; i < humanWords.length; i++) {
      if (text === humanWords[i]) return i;
    }
    return -1;
  }

  return {
    name: 'にんげん(村の長老)',
    vocabSize: humanWords.length,
    speak,
    understand,
  };
}

/**
 * 純ニューラル50音通訳脳（辞書ゼロ・Float32Array のみ）:
 * 2文字の50音ID列 [t0, t1] → 隠れ層 H (tanh) → 変換後の2文字50音ID列 [o0, o1, EOS]
 */
export function createNeuralKanaTranslatorV16(opts = {}) {
  const { name = 'つうやくちご(純ニューラル)', hiddenSize = 48, seed = 1601 } = opts;
  const rng = createRng(seed);
  const H = hiddenSize;
  const IN_DIM = VOCAB_SIZE * 2;

  const init = (len, s) => {
    const a = new Float32Array(len);
    for (let i = 0; i < len; i++) a[i] = (rng() * 2 - 1) * s;
    return a;
  };

  const W1 = init(H * IN_DIM, 0.35);
  const b1 = new Float32Array(H);
  const Wo1 = init(KANA_COUNT * H, 0.25);
  const bo1 = new Float32Array(KANA_COUNT);
  const Wo2 = init(KANA_COUNT * H, 0.25);
  const bo2 = new Float32Array(KANA_COUNT);

  const h = new Float32Array(H);
  const l1 = new Float32Array(KANA_COUNT);
  const p1 = new Float32Array(KANA_COUNT);
  const l2 = new Float32Array(KANA_COUNT);
  const p2 = new Float32Array(KANA_COUNT);

  function forward(srcIds) {
    const t0 = srcIds[0] ?? 0;
    const t1 = srcIds[1] ?? 0;
    for (let i = 0; i < H; i++) {
      const row = i * IN_DIM;
      h[i] = Math.tanh(b1[i] + W1[row + t0] + W1[row + VOCAB_SIZE + t1]);
    }
    for (let k = 0; k < KANA_COUNT; k++) {
      const row = k * H;
      let s1 = bo1[k];
      let s2 = bo2[k];
      for (let i = 0; i < H; i++) {
        s1 += Wo1[row + i] * h[i];
        s2 += Wo2[row + i] * h[i];
      }
      l1[k] = s1;
      l2[k] = s2;
    }
    softmaxArr(l1, p1);
    softmaxArr(l2, p2);
    return { t0, t1 };
  }

  function trainPair(srcIds, dstIds, lr = 0.08) {
    const { t0, t1 } = forward(srcIds);
    const k1 = (dstIds[0] ?? KANA_START) - KANA_START;
    const k2 = (dstIds[1] ?? KANA_START) - KANA_START;
    const dh = new Float32Array(H);

    for (let k = 0; k < KANA_COUNT; k++) {
      const e1 = p1[k] - (k === k1 ? 1 : 0);
      const e2 = p2[k] - (k === k2 ? 1 : 0);
      bo1[k] -= lr * e1;
      bo2[k] -= lr * e2;
      const row = k * H;
      for (let i = 0; i < H; i++) {
        dh[i] += Wo1[row + i] * e1 + Wo2[row + i] * e2;
        Wo1[row + i] -= lr * e1 * h[i];
        Wo2[row + i] -= lr * e2 * h[i];
      }
    }

    for (let i = 0; i < H; i++) {
      const dt = dh[i] * (1 - h[i] * h[i]);
      b1[i] -= lr * dt;
      const row = i * IN_DIM;
      W1[row + t0] -= lr * dt;
      W1[row + VOCAB_SIZE + t1] -= lr * dt;
    }
  }

  function translate(srcIds) {
    forward(srcIds);
    const o1 = KANA_START + argmaxArr(p1);
    const o2 = KANA_START + argmaxArr(p2);
    return Uint8Array.from([o1, o2, TOKEN_EOS]);
  }

  return { name, trainPair, translate };
}
