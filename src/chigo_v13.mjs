// chigo_v13.mjs — 純ニューラルちご v13「無教師・報酬サバイバル脳 (50音 Double-DQN)」
// 100%純ニューラル・ゼロ依存・厳密50音I/O (0..49)
// 観測テキストのdecode・文字列マッチ・生存ルール記述・方策教師は一切ゼロ！
//
// v12からの大進化（AGIへの第3の壁「無教師・身体性サバイバル」を突破）:
//   1. 観測は生の50音ID列 (Uint8Array 0..49) のみ。日本語へデコードせず、
//      ユニグラム(50)＋決定論的バイグラムハッシュ(512)のスパース50音特徴から
//      2層ニューラルQネットワーク (562 → 64 ReLU → 48 ReLU → 6) へ直接入力。
//   2. 行動の正解を誰にも教わらず、固定シード縄文サバイバル環境 (jomon_world.mjs) の
//      生死と空腹のスカラ報酬 (reward) だけから Double-DQN + 経験リプレイ + ターゲット網 + Adam で
//      春夏秋冬のサバイバル戦略（秋のどんぐり備蓄・冬の火維持・空腹時の食事・疲労時の休息）を自力獲得する！

import { TOKEN_SEP, TOKEN_EOS, encode50 } from './codec50.mjs';
import { createRng } from './jomon_world.mjs';

const MOTOR_UTTERANCES = [
  'ひをたす。',   // fire
  'もりひろう。', // gather
  'さかなつる。', // fish
  'しかかり。',   // hunt
  'くう。',       // eat
  'ねる。',       // rest
];
const MOTOR_IDS = MOTOR_UTTERANCES.map((u) => encode50(u, { appendEos: true }));
const NUM_ACTIONS = MOTOR_IDS.length;

const SEG_BUCKETS = 256;
const PAIR_BUCKETS = 512;
const IN_DIM = SEG_BUCKETS + PAIR_BUCKETS; // 768

/**
 * 生の50音ID列 (0..49) を文字列化せず、TOKEN_SEP(・)区切りの純粋なトークン列ハッシュ＋
 * ペア相互作用ハッシュへ直接変換する（日本語キーワードやルールは一切ゼロ）。
 */
function extractSparse50(obsIds) {
  const segHashes = [];
  let h = 2166136261 >>> 0;
  let hasTok = false;
  for (let i = 0; i < obsIds.length; i++) {
    const t = obsIds[i];
    if (t === TOKEN_SEP || t === TOKEN_EOS) {
      if (hasTok) {
        segHashes.push(h >>> 0);
        h = 2166136261 >>> 0;
        hasTok = false;
      }
    } else {
      h = Math.imul(h ^ (t + 1), 16777619) >>> 0;
      hasTok = true;
    }
  }
  if (hasTok) segHashes.push(h >>> 0);

  const seen = new Uint8Array(IN_DIM);
  const idxs = [];
  for (let i = 0; i < segHashes.length; i++) {
    const b = segHashes[i] & (SEG_BUCKETS - 1);
    if (!seen[b]) {
      seen[b] = 1;
      idxs.push(b);
    }
  }
  for (let i = 0; i < segHashes.length; i++) {
    for (let j = i + 1; j < segHashes.length; j++) {
      const ph = Math.imul(segHashes[i] ^ 0x9e3779b9, 2654435761) ^ segHashes[j];
      const b = SEG_BUCKETS + ((ph >>> 0) & (PAIR_BUCKETS - 1));
      if (!seen[b]) {
        seen[b] = 1;
        idxs.push(b);
      }
    }
  }
  return Int32Array.from(idxs);
}

export function createSurvivalChigoV13(opts = {}) {
  const {
    name = 'ちごv13(純ニューラルRL)',
    h1Size = 64,
    h2Size = 48,
    seed = 1337,
    gamma = 0.96,
    lr = 0.0025,
    batchSize = 24,
    bufferSize = 8000,
    targetSyncEvery = 160,
  } = opts;

  let rng = createRng(seed);
  const H1 = h1Size;
  const H2 = h2Size;

  function makeNet() {
    const s1 = Math.sqrt(2 / 60);
    const s2 = Math.sqrt(2 / H1);
    const s3 = Math.sqrt(1 / H2) * 0.5;
    const init = (len, scale) => {
      const a = new Float32Array(len);
      for (let i = 0; i < len; i++) a[i] = (rng() * 2 - 1) * scale;
      return a;
    };
    return {
      W1: init(H1 * IN_DIM, s1),
      b1: new Float32Array(H1),
      W2: init(H2 * H1, s2),
      b2: new Float32Array(H2),
      W3: init(NUM_ACTIONS * H2, s3),
      b3: new Float32Array(NUM_ACTIONS),
    };
  }

  const net = makeNet();
  const targetNet = makeNet();
  const keys = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3'];

  function syncTarget() {
    for (const k of keys) targetNet[k].set(net[k]);
  }
  syncTarget();

  const grads = {};
  const adamM = {};
  const adamV = {};
  for (const k of keys) {
    grads[k] = new Float32Array(net[k].length);
    adamM[k] = new Float32Array(net[k].length);
    adamV[k] = new Float32Array(net[k].length);
  }
  let adamStep = 0;

  // 経験リプレイバッファ
  const replay = [];
  let replayPos = 0;

  let evalMode = false;
  let totalSteps = 0;
  let lastActionIdx = 0;
  let lastSparse = null;

  function forward(model, sparseIdxs) {
    const h1 = new Float32Array(H1);
    for (let i = 0; i < H1; i++) {
      const row = i * IN_DIM;
      let s = model.b1[i];
      for (let k = 0; k < sparseIdxs.length; k++) s += model.W1[row + sparseIdxs[k]];
      h1[i] = s > 0 ? s : 0;
    }
    const h2 = new Float32Array(H2);
    for (let i = 0; i < H2; i++) {
      const row = i * H1;
      let s = model.b2[i];
      for (let j = 0; j < H1; j++) s += model.W2[row + j] * h1[j];
      h2[i] = s > 0 ? s : 0;
    }
    const q = new Float32Array(NUM_ACTIONS);
    for (let a = 0; a < NUM_ACTIONS; a++) {
      const row = a * H2;
      let s = model.b3[a];
      for (let j = 0; j < H2; j++) s += model.W3[row + j] * h2[j];
      q[a] = s;
    }
    return { h1, h2, q };
  }

  function argmax(arr) {
    let bestIdx = 0;
    let bestVal = arr[0];
    for (let i = 1; i < arr.length; i++) {
      if (arr[i] > bestVal) {
        bestVal = arr[i];
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  function trainBatch() {
    if (replay.length < batchSize * 4) return;
    for (const k of keys) grads[k].fill(0);

    const invB = 1 / batchSize;
    const dh2 = new Float32Array(H2);
    const dh1 = new Float32Array(H1);

    for (let b = 0; b < batchSize; b++) {
      const tr = replay[Math.floor(rng() * replay.length)];
      const { h1, h2, q } = forward(net, tr.s);

      // Double-DQN ターゲット
      let target = tr.r;
      if (!tr.done) {
        const nextOnline = forward(net, tr.ns).q;
        const bestNextA = argmax(nextOnline);
        const nextTargetQ = forward(targetNet, tr.ns).q[bestNextA];
        target += gamma * nextTargetQ;
      }

      // Huber 損失の勾配 (外れ値報酬で発散しないように)
      const diff = q[tr.a] - target;
      const dQ = Math.max(-1.5, Math.min(1.5, diff)) * invB;

      // W3, b3 の勾配 & dh2
      grads.b3[tr.a] += dQ;
      const w3Row = tr.a * H2;
      for (let j = 0; j < H2; j++) {
        grads.W3[w3Row + j] += dQ * h2[j];
        dh2[j] = h2[j] > 0 ? net.W3[w3Row + j] * dQ : 0;
      }

      // W2, b2 の勾配 & dh1
      dh1.fill(0);
      for (let i = 0; i < H2; i++) {
        const g2 = dh2[i];
        if (g2 === 0) continue;
        grads.b2[i] += g2;
        const w2Row = i * H1;
        for (let j = 0; j < H1; j++) {
          grads.W2[w2Row + j] += g2 * h1[j];
          dh1[j] += net.W2[w2Row + j] * g2;
        }
      }

      // W1, b1 の勾配 (スパース)
      for (let i = 0; i < H1; i++) {
        if (h1[i] <= 0) continue;
        const g1 = dh1[i];
        if (g1 === 0) continue;
        grads.b1[i] += g1;
        const w1Row = i * IN_DIM;
        for (let k = 0; k < tr.s.length; k++) {
          grads.W1[w1Row + tr.s[k]] += g1;
        }
      }
    }

    // Adam 更新
    adamStep += 1;
    const b1 = 0.9;
    const b2 = 0.999;
    const eps = 1e-8;
    const bc1 = 1 - Math.pow(b1, adamStep);
    const bc2 = 1 - Math.pow(b2, adamStep);
    const wd = 0.00005;

    for (const k of keys) {
      const p = net[k];
      const g = grads[k];
      const m = adamM[k];
      const v = adamV[k];
      const isBias = k.startsWith('b');
      for (let i = 0; i < p.length; i++) {
        const gi = g[i] + (isBias ? 0 : wd * p[i]);
        m[i] = b1 * m[i] + (1 - b1) * gi;
        v[i] = b2 * v[i] + (1 - b2) * gi * gi;
        p[i] -= (lr * (m[i] / bc1)) / (Math.sqrt(v[i] / bc2) + eps);
      }
    }
  }

  function setEvalMode(flag) {
    evalMode = Boolean(flag);
  }

  function reset() {
    lastSparse = null;
  }

  function act(obsIds) {
    const sp = extractSparse50(obsIds);
    lastSparse = sp;

    let actionIdx;
    if (!evalMode) {
      // ε-greedy 探索（序盤は広く試し、中盤以降は秋・冬まで到達して四季を経験する）
      const eps = Math.max(0.05, 0.95 * Math.exp(-totalSteps / 1100));
      if (rng() < eps) {
        actionIdx = Math.floor(rng() * NUM_ACTIONS);
      } else {
        actionIdx = argmax(forward(net, sp).q);
      }
    } else {
      actionIdx = argmax(forward(net, sp).q);
    }

    lastActionIdx = actionIdx;
    return MOTOR_IDS[actionIdx];
  }

  function learn(obsIds, outIds, reward, nextObsIds, done) {
    const s = lastSparse || extractSparse50(obsIds);
    const ns = extractSparse50(nextObsIds);
    const item = { s, a: lastActionIdx, r: reward, ns, done };

    if (replay.length < bufferSize) replay.push(item);
    else {
      replay[replayPos] = item;
      replayPos = (replayPos + 1) % bufferSize;
    }

    totalSteps += 1;
    if (totalSteps % 2 === 0) trainBatch();
    if (totalSteps % targetSyncEvery === 0) syncTarget();
  }

  function countParams() {
    let n = 0;
    for (const k of keys) n += net[k].length;
    return n;
  }

  return {
    name,
    setEvalMode,
    reset,
    act,
    learn,
    countParams,
  };
}
