// chigo_v15.mjs — 純ニューラルちご v15「創発言語・2体のちごによる50音プロトコル自力発明＆文化伝承」
// 100%純ニューラル・ゼロ依存・厳密50音I/O (0..49)
// 初期語彙ゼロ・教師データゼロ・エージェント間の勾配共有ゼロ！
//
// v14からの大進化（個体知能 → 社会知能・ことばの起源）:
//   1. 話し手(あに) と 聞き手(おとうと) の2体の無知な純ニューラルちごが、
//      「話し手だけが見ている縄文世界の状況」を、厳密な50音2文字 (Uint8Array 0..49) のみで伝え合う。
//   2. 誰も単語を教えない。2体が協力成功したときだけ共通の生存報酬 (+1) を得る
//      完全分散型 REINFORCE（ベースライン＋エントロピー正則化）だけで、
//      2体の間に「独自の50音ことば（創発語彙）」が自然発生する！
//   3. 第1世代が発明したことばを、生まれたての第2世代へ50音通信だけで引き継ぐ
//      「文化伝承（Iterated Learning）」と、3×3構成的状況からの未経験組み合わせ
//      ゼロショット解読も検証できる。

import { VOCAB_SIZE, TOKEN_EOS, decode50 } from './codec50.mjs';
import { createRng } from './jomon_world.mjs';

const KANA_START = 4; // 清音46文字: ID 4..49
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

function sampleCategorical(probs, rng) {
  const u = rng();
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (u <= acc) return i;
  }
  return probs.length - 1;
}

function argmaxArr(arr) {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

/**
 * 話し手ちご（Speaker）:
 * 非言語の観測ベクトル x (dim = obsDim) を受け取り、
 * 2文字の50音ID列 [c1, c2, TOKEN_EOS] を純ニューラル方策で発声する。
 */
export function createSpeakerChigoV15(opts = {}) {
  const { name = 'あに(話し手)', obsDim = 6, hiddenSize = 32, seed = 1501 } = opts;
  const rng = createRng(seed);
  const H = hiddenSize;

  const init = (len, s) => {
    const a = new Float32Array(len);
    for (let i = 0; i < len; i++) a[i] = (rng() * 2 - 1) * s;
    return a;
  };

  // obsDim -> H (tanh) -> 1文字目 (46) & 2文字目 (46, 1文字目埋め込み条件付き)
  const W1 = init(H * obsDim, 0.4);
  const b1 = new Float32Array(H);
  const Wc1 = init(KANA_COUNT * H, 0.2);
  const bc1 = new Float32Array(KANA_COUNT);
  const Wc2 = init(KANA_COUNT * H, 0.2);
  const Wc2Prev = init(KANA_COUNT * KANA_COUNT, 0.15);
  const bc2 = new Float32Array(KANA_COUNT);

  const h = new Float32Array(H);
  const l1 = new Float32Array(KANA_COUNT);
  const p1 = new Float32Array(KANA_COUNT);
  const l2 = new Float32Array(KANA_COUNT);
  const p2 = new Float32Array(KANA_COUNT);

  let lastObs = null;
  let lastK1 = 0;
  let lastK2 = 0;

  function speak(obsVec, greedy = false) {
    lastObs = obsVec;
    for (let i = 0; i < H; i++) {
      const row = i * obsDim;
      let s = b1[i];
      for (let j = 0; j < obsDim; j++) s += W1[row + j] * obsVec[j];
      h[i] = Math.tanh(s);
    }

    for (let k = 0; k < KANA_COUNT; k++) {
      const row = k * H;
      let s = bc1[k];
      for (let i = 0; i < H; i++) s += Wc1[row + i] * h[i];
      l1[k] = s;
    }
    softmaxArr(l1, p1);
    const k1 = greedy ? argmaxArr(p1) : sampleCategorical(p1, rng);

    for (let k = 0; k < KANA_COUNT; k++) {
      const row = k * H;
      let s = bc2[k] + Wc2Prev[k * KANA_COUNT + k1];
      for (let i = 0; i < H; i++) s += Wc2[row + i] * h[i];
      l2[k] = s;
    }
    softmaxArr(l2, p2);
    const k2 = greedy ? argmaxArr(p2) : sampleCategorical(p2, rng);

    lastK1 = k1;
    lastK2 = k2;
    return Uint8Array.from([KANA_START + k1, KANA_START + k2, TOKEN_EOS]);
  }

  function reinforce(advantage, { lr = 0.045, entropyCoef = 0.015 } = {}) {
    const dh = new Float32Array(H);

    // 1文字目 & 2文字目の REINFORCE + エントロピー勾配
    for (let k = 0; k < KANA_COUNT; k++) {
      const pk1 = p1[k];
      const ent1 = -pk1 * (Math.log(Math.max(1e-7, pk1)) + 1);
      const g1 = ((k === lastK1 ? 1 : 0) - pk1) * advantage + entropyCoef * ent1;
      bc1[k] += lr * g1;
      const r1 = k * H;
      for (let i = 0; i < H; i++) {
        dh[i] += Wc1[r1 + i] * g1;
        Wc1[r1 + i] += lr * g1 * h[i];
      }

      const pk2 = p2[k];
      const ent2 = -pk2 * (Math.log(Math.max(1e-7, pk2)) + 1);
      const g2 = ((k === lastK2 ? 1 : 0) - pk2) * advantage + entropyCoef * ent2;
      bc2[k] += lr * g2;
      Wc2Prev[k * KANA_COUNT + lastK1] += lr * g2;
      const r2 = k * H;
      for (let i = 0; i < H; i++) {
        dh[i] += Wc2[r2 + i] * g2;
        Wc2[r2 + i] += lr * g2 * h[i];
      }
    }

    for (let i = 0; i < H; i++) {
      const dt = dh[i] * (1 - h[i] * h[i]);
      b1[i] += lr * dt;
      const row = i * obsDim;
      for (let j = 0; j < obsDim; j++) {
        if (lastObs[j] !== 0) W1[row + j] += lr * dt * lastObs[j];
      }
    }
  }

  /**
   * 模倣発声（親や村の長老の50音を聞いてマネする）
   */
  function imitate(obsVec, targetIds, lr = 0.06) {
    speak(obsVec, true);
    const t1 = (targetIds[0] ?? KANA_START) - KANA_START;
    const t2 = (targetIds[1] ?? KANA_START) - KANA_START;
    const dh = new Float32Array(H);

    for (let k = 0; k < KANA_COUNT; k++) {
      const g1 = (k === t1 ? 1 : 0) - p1[k];
      bc1[k] += lr * g1;
      const r1 = k * H;
      for (let i = 0; i < H; i++) {
        dh[i] += Wc1[r1 + i] * g1;
        Wc1[r1 + i] += lr * g1 * h[i];
      }

      const g2 = (k === t2 ? 1 : 0) - p2[k];
      bc2[k] += lr * g2;
      Wc2Prev[k * KANA_COUNT + t1] += lr * g2;
      const r2 = k * H;
      for (let i = 0; i < H; i++) {
        dh[i] += Wc2[r2 + i] * g2;
        Wc2[r2 + i] += lr * g2 * h[i];
      }
    }

    for (let i = 0; i < H; i++) {
      const dt = dh[i] * (1 - h[i] * h[i]);
      b1[i] += lr * dt;
      const row = i * obsDim;
      for (let j = 0; j < obsDim; j++) {
        if (obsVec[j] !== 0) W1[row + j] += lr * dt * obsVec[j];
      }
    }
  }

  return { name, speak, reinforce, imitate };
}

/**
 * 聞き手ちご（Listener）:
 * 話し手が発した50音ID列 (Uint8Array 0..49) だけを聴取し、
 * 行動ヘッド群（単一行動または複数属性ヘッド）から行動をサンプリングする。
 */
export function createListenerChigoV15(opts = {}) {
  const { name = 'おとうと(聞き手)', headDims = [6], hiddenSize = 40, seed = 1502 } = opts;
  const rng = createRng(seed);
  const H = hiddenSize;
  const IN_DIM = VOCAB_SIZE * 2; // 1文字目(50) + 2文字目(50)

  const init = (len, s) => {
    const a = new Float32Array(len);
    for (let i = 0; i < len; i++) a[i] = (rng() * 2 - 1) * s;
    return a;
  };

  const W1 = init(H * IN_DIM, 0.35);
  const b1 = new Float32Array(H);
  const heads = headDims.map((dim) => ({
    dim,
    W: init(dim * H, 0.25),
    b: new Float32Array(dim),
    logits: new Float32Array(dim),
    probs: new Float32Array(dim),
    lastAct: 0,
  }));

  const h = new Float32Array(H);
  let lastT0 = 0;
  let lastT1 = 0;

  function listenAndAct(msgIds, greedy = false) {
    const t0 = msgIds[0] ?? 0;
    const t1 = msgIds[1] ?? 0;
    lastT0 = t0;
    lastT1 = t1;

    for (let i = 0; i < H; i++) {
      const row = i * IN_DIM;
      h[i] = Math.tanh(b1[i] + W1[row + t0] + W1[row + VOCAB_SIZE + t1]);
    }

    const actions = [];
    for (const hd of heads) {
      for (let a = 0; a < hd.dim; a++) {
        const row = a * H;
        let s = hd.b[a];
        for (let i = 0; i < H; i++) s += hd.W[row + i] * h[i];
        hd.logits[a] = s;
      }
      softmaxArr(hd.logits, hd.probs);
      const act = greedy ? argmaxArr(hd.probs) : sampleCategorical(hd.probs, rng);
      hd.lastAct = act;
      actions.push(act);
    }
    return actions;
  }

  function reinforce(advantage, { lr = 0.055, entropyCoef = 0.01 } = {}) {
    const dh = new Float32Array(H);
    for (const hd of heads) {
      for (let a = 0; a < hd.dim; a++) {
        const pa = hd.probs[a];
        const ent = -pa * (Math.log(Math.max(1e-7, pa)) + 1);
        const g = ((a === hd.lastAct ? 1 : 0) - pa) * advantage + entropyCoef * ent;
        hd.b[a] += lr * g;
        const row = a * H;
        for (let i = 0; i < H; i++) {
          dh[i] += hd.W[row + i] * g;
          hd.W[row + i] += lr * g * h[i];
        }
      }
    }

    for (let i = 0; i < H; i++) {
      const dt = dh[i] * (1 - h[i] * h[i]);
      b1[i] += lr * dt;
      const row = i * IN_DIM;
      W1[row + lastT0] += lr * dt;
      W1[row + VOCAB_SIZE + lastT1] += lr * dt;
    }
  }

  return { name, listenAndAct, reinforce };
}

/**
 * 2体のちごによる「創発50音語彙」の対話辞書を覗き見るユーティリティ
 */
export function inspectEmergentLexicon(speaker, listener, scenarios) {
  return scenarios.map((sc) => {
    const msgIds = speaker.speak(sc.obs, true);
    const acts = listener.listenAndAct(msgIds, true);
    const ok = sc.targetActs.every((t, i) => acts[i] === t);
    return {
      状況: sc.label,
      発明した50音ことば: decode50(msgIds),
      聞き手の理解: ok ? '✓ 通じた' : '✗ 誤解',
      ok,
    };
  });
}
