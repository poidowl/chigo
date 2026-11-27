// jomon_baby_v1.mjs — 非LLM・発達型 原始人AGI (v1)
// 制約:
//   1. 入出力は codec50.mjs の50音ID列 (0..49) のみ
//   2. 出力語彙は 1〜2文字のヤマトコトバのみ（3文字以上は完全封印）
//   3. 赤ん坊のように段階発達する:
//      Stage 0: 一語文（短い1〜2文字語の連発）
//      Stage 1: 二語文（対象・状態 / 対象・行動 の2語ペア）
//      Stage 2: 二語文の因果連鎖＋未知語の対話学習（原始人AGI）

import { encode50, decode50, isValid50 } from './codec50.mjs';

// 1〜2文字の初期ヤマトコトバ辞書（3文字以上は封印）
// kind: 'thing' (もの/自然), 'act' (うごき), 'state' (ようす), 'ask' (とい)
const SEED_WORDS = {
  // 1文字語
  'ひ': 'thing', 'き': 'thing', 'み': 'thing', 'め': 'thing',
  'て': 'thing', 'ち': 'thing', 'ね': 'thing', 'は': 'thing',
  'よ': 'thing', 'え': 'thing',
  // 2文字語（自然・生き物・体）
  'みず': 'thing', 'あめ': 'thing', 'かぜ': 'thing', 'やま': 'thing',
  'うみ': 'thing', 'かわ': 'thing', 'いし': 'thing', 'つち': 'thing',
  'くさ': 'thing', 'はな': 'thing', 'とり': 'thing', 'うお': 'thing',
  'しか': 'thing', 'くま': 'thing', 'ひと': 'thing', 'はら': 'thing',
  'うえ': 'thing', 'した': 'thing', 'そら': 'thing', 'もり': 'thing',
  // 2文字語（うごき・状態・問い）
  'あつ': 'state', 'さむ': 'state', 'こわ': 'state', 'うま': 'state',
  'いた': 'state', 'よい': 'state', 'やだ': 'state', 'ある': 'state',
  'ない': 'state', 'へる': 'state', 'きえ': 'act', 'もえ': 'act',
  'くる': 'act', 'ゆく': 'act', 'くう': 'act', 'のむ': 'act',
  'ねる': 'act', 'にげ': 'act', 'みる': 'act', 'きく': 'act',
  'もつ': 'act', 'かく': 'act', 'たす': 'act', 'とる': 'act',
  'なに': 'ask', 'なぜ': 'ask', 'どこ': 'ask', 'にて': 'ask',
};

// 原始人の体験則（初期因果・連想エッジ: [from, to, relation, weight]）
// relation: 'cause' (引き起こす), 'attr' (特徴), 'action' (とるべき動き)
const SEED_EDGES = [
  ['ひ', 'あつ', 'attr', 1.0],
  ['ひ', 'もえ', 'attr', 0.9],
  ['ひ', 'き', 'action', 0.85],     // 火→木(を足す)
  ['き', 'もえ', 'cause', 0.8],
  ['あめ', 'みず', 'attr', 0.9],
  ['あめ', 'ひ', 'cause', 0.95],    // 雨→火(が消える)
  ['あめ', 'きえ', 'cause', 0.95],
  ['あめ', 'さむ', 'cause', 0.8],
  ['あめ', 'やだ', 'attr', 0.7],
  ['かぜ', 'さむ', 'cause', 0.85],
  ['かぜ', 'きえ', 'cause', 0.7],
  ['くま', 'こわ', 'attr', 1.0],
  ['くま', 'にげ', 'action', 1.0],
  ['しか', 'うま', 'attr', 0.9],
  ['しか', 'とる', 'action', 0.85],
  ['しか', 'くう', 'action', 0.8],
  ['うお', 'かわ', 'attr', 0.9],
  ['うお', 'うま', 'attr', 0.85],
  ['うお', 'くう', 'action', 0.8],
  ['み', 'もり', 'attr', 0.85],
  ['み', 'くう', 'action', 0.85],
  ['みず', 'のむ', 'action', 0.95],
  ['みず', 'かわ', 'attr', 0.8],
  ['はら', 'くう', 'action', 0.95],
  ['はら', 'み', 'action', 0.8],
  ['へる', 'くう', 'action', 1.0],
  ['さむ', 'ひ', 'action', 1.0],
  ['さむ', 'たす', 'action', 0.85],
  ['よ', 'ねる', 'action', 0.85],
  ['よ', 'ひ', 'action', 0.8],
  ['いし', 'もつ', 'action', 0.7],
  ['いし', 'いた', 'cause', 0.7],
];

/**
 * 1〜2文字語だけを切り出す簡易セグメンタ（最長2文字マッチ＋区切り優先）。
 * 未知の長い語は先頭2文字の原始語根に圧縮する（3文字以上は封印）。
 */
export function segmentShortWords(text, knownVocab) {
  const chunks = text
    .replace(/。/g, '・')
    .split('・')
    .map((s) => s.trim())
    .filter(Boolean);

  const tokens = [];
  for (const chunk of chunks) {
    // すでに1〜2文字ならそのまま採用（未知語でも1〜2文字として認識）
    if (chunk.length <= 2) {
      tokens.push(chunk);
      continue;
    }
    // 3文字以上の塊は既知の2文字→1文字で切り出し、残りは2文字単位で原始語根化
    let i = 0;
    while (i < chunk.length) {
      const two = chunk.slice(i, i + 2);
      const one = chunk.slice(i, i + 1);
      if (two.length === 2 && knownVocab.has(two)) {
        tokens.push(two);
        i += 2;
      } else if (knownVocab.has(one)) {
        tokens.push(one);
        i += 1;
      } else {
        // 未知語は最大2文字に制限（3文字以上は封印）
        tokens.push(two);
        i += two.length;
      }
    }
  }
  return tokens;
}

/**
 * 出力単語が厳密に1〜2文字（濁音合成後）であることを保証する。
 */
function enforceMax2Chars(word) {
  return word.length <= 2 ? word : word.slice(0, 2);
}

export function createJomonBabyV1(opts = {}) {
  const { startStage = 0 } = opts;

  const vocab = new Map(Object.entries(SEED_WORDS));
  // graph: Map<fromWord, Map<toWord, { rel, w }>>
  const graph = new Map();
  const episodes = []; // 過去の対話記憶（直近の経験）

  let experience = startStage === 0 ? 0 : (startStage === 1 ? 4 : 10);
  let lastUnknownWord = null; // 直前に聞き返した未知語

  function addEdge(from, to, rel = 'assoc', delta = 0.6) {
    const f = enforceMax2Chars(from);
    const t = enforceMax2Chars(to);
    if (!f || !t || f === t) return;
    if (!graph.has(f)) graph.set(f, new Map());
    const neighbors = graph.get(f);
    const cur = neighbors.get(t);
    if (cur) {
      cur.w = Math.min(2.0, cur.w + delta);
      if (rel !== 'assoc') cur.rel = rel;
    } else {
      neighbors.set(t, { rel, w: delta });
    }
  }

  for (const [f, t, rel, w] of SEED_EDGES) {
    addEdge(f, t, rel, w);
  }

  function getStage() {
    if (experience < 4) return 0; // 赤ん坊：一語文（短語連発）
    if (experience < 10) return 1; // 幼児：二語文
    return 2; // 原始人AGI：二語文の因果連鎖＋未知語学習
  }

  function getTopNeighbors(word, relFilter = null, limit = 3) {
    const m = graph.get(word);
    if (!m) return [];
    const arr = [];
    for (const [to, info] of m.entries()) {
      if (relFilter && info.rel !== relFilter) continue;
      arr.push({ word: to, rel: info.rel, w: info.w });
    }
    arr.sort((a, b) => b.w - a.w);
    return arr.slice(0, limit);
  }

  function formatUtterance(clauses) {
    // 各句は [w1] または [w1, w2]（すべて1〜2文字に制限）
    const clean = clauses
      .map((pair) => pair.map(enforceMax2Chars).filter(Boolean).join('・'))
      .filter(Boolean);
    if (clean.length === 0) return 'う。';
    return clean.join('。') + '。';
  }

  /**
   * 厳密50音I/O: Uint8Array(0..49) -> Uint8Array(0..49)
   */
  function think50(inputIds) {
    if (!isValid50(inputIds)) {
      return encode50('なに。', { appendEos: true });
    }
    const text = decode50(inputIds);
    const words = segmentShortWords(text, vocab);
    const stage = getStage();

    // 1) 直前に質問した未知語への「教え」を受け取った場合 → グラフに記憶！
    const learnedNow = [];
    if (lastUnknownWord && words.length > 0) {
      const target = lastUnknownWord;
      const teachingWords = words.filter((w) => w !== target && w !== 'なに');
      if (teachingWords.length > 0) {
        if (!vocab.has(target)) vocab.set(target, 'thing');
        for (const tw of teachingWords) {
          if (!vocab.has(tw)) vocab.set(tw, 'thing');
          addEdge(target, tw, 'attr', 1.1);
          addEdge(tw, target, 'assoc', 0.6);
          learnedNow.push(tw);
        }
        lastUnknownWord = null;
        experience += 2;
      }
    }

    // 2) 入力内の未知語を検出（ただし教わった直後は除く）
    const unknownWords = words.filter((w) => !vocab.has(w));
    const knownWords = words.filter((w) => vocab.has(w));

    // 共起エッジを弱く自動学習（赤ん坊の連合学習）
    for (let i = 0; i < words.length; i++) {
      for (let j = i + 1; j < words.length; j++) {
        if (vocab.has(words[i]) && vocab.has(words[j])) {
          addEdge(words[i], words[j], 'assoc', 0.15);
          addEdge(words[j], words[i], 'assoc', 0.15);
        }
      }
    }

    episodes.push({ inWords: words, stage });
    if (episodes.length > 50) episodes.shift();

    let clauses = [];

    // 3) 教わった直後の応答（覚えたことを自分の口で確認＝AGI臭さ）
    if (learnedNow.length > 0) {
      const target = episodes.length >= 2
        ? episodes[episodes.length - 2].inWords.find((w) => vocab.has(w)) || words[0]
        : words[0];
      const top = learnedNow[0];
      if (stage === 0) {
        clauses.push([top]);
      } else {
        clauses.push([target, top]);
        clauses.push([top, 'よい']);
      }
      const replyText = formatUtterance(clauses);
      return encode50(replyText, { appendEos: true });
    }

    // 4) 未知語が来た場合 → 覚えて問い返す
    if (unknownWords.length > 0) {
      const unk = enforceMax2Chars(unknownWords[0]);
      lastUnknownWord = unk;
      experience += 1;

      if (stage === 0) {
        // 赤ん坊: 単語だけオウム返し＋「なに」
        clauses.push([unk]);
        clauses.push(['なに']);
      } else if (stage === 1) {
        // 幼児: 二語文で質問
        clauses.push([unk, 'なに']);
      } else {
        // 原始人AGI: 未知語を質問しつつ、文脈の既知語と似てるか仮説を投げる
        clauses.push([unk, 'なに']);
        if (knownWords.length > 0) {
          clauses.push([knownWords[0], 'にて']);
        } else {
          clauses.push(['ひ', 'にて']);
        }
      }
      const replyText = formatUtterance(clauses);
      return encode50(replyText, { appendEos: true });
    }

    // 5) 既知語からの推論（ステージごとに表現が発達）
    experience += 1;
    const primary = knownWords[0] || 'ひ';
    const causes = getTopNeighbors(primary, 'cause', 2);
    const actions = getTopNeighbors(primary, 'action', 2);
    const attrs = getTopNeighbors(primary, 'attr', 2);
    const anyAssoc = getTopNeighbors(primary, null, 3);

    // 入力の2語目（例: 「はら・へる」の「へる」）からも行動を引く
    const secondary = knownWords[1];
    const secActions = secondary ? getTopNeighbors(secondary, 'action', 2) : [];

    if (stage === 0) {
      // Stage 0（赤ん坊）: 1〜2文字の単語をポツポツ連発（一語文）
      clauses.push([primary]);
      const firstHit = causes[0] || actions[0] || attrs[0] || anyAssoc[0];
      if (firstHit) clauses.push([firstHit.word]);
    } else if (stage === 1) {
      // Stage 1（幼児）: 二語文（対象＋因果 or 行動）
      const hit = causes[0] || secActions[0] || actions[0] || attrs[0] || anyAssoc[0];
      if (hit) {
        clauses.push([primary, hit.word]);
      } else {
        clauses.push([primary, 'ある']);
      }
    } else {
      // Stage 2（原始人AGI）: 観察/予測（因果） → 行動提案 の二語文連鎖
      // 例: 「あめ・くる」→ 「ひ・きえ。き・かく。」
      if (causes.length >= 2) {
        clauses.push([causes[0].word, causes[1].word]);
      } else if (causes.length === 1) {
        clauses.push([primary, causes[0].word]);
      } else if (attrs.length >= 1) {
        clauses.push([primary, attrs[0].word]);
      }

      const bestAct = secActions[0] || actions[0];
      if (bestAct) {
        // 行動の対象（例: さむ→ひ・たす、くま→ひと・にげ）
        const actTarget = (bestAct.word === 'にげ') ? 'ひと'
          : (bestAct.word === 'くう') ? (actions.find((a) => a.word === 'み')?.word || 'み')
          : primary;
        clauses.push([actTarget, bestAct.word]);
      } else if (causes.length > 0) {
        // 2ホップ先（因果の先にある対処行動）をグラフ探索！
        const hop2 = getTopNeighbors(causes[0].word, 'action', 1)[0];
        if (hop2) {
          clauses.push([causes[0].word, hop2.word]);
        }
      }

      if (clauses.length === 0) {
        clauses.push([primary, anyAssoc[0]?.word || 'ある']);
      }
    }

    const replyText = formatUtterance(clauses);
    return encode50(replyText, { appendEos: true });
  }

  return {
    name: 'v1-jomon-baby',
    getStage,
    getExperience: () => experience,
    getVocabSize: () => vocab.size,
    think50,
    talk(yamatoText) {
      const inIds = encode50(yamatoText, { appendEos: true });
      const outIds = think50(inIds);
      return {
        inIds,
        outIds,
        reply: decode50(outIds),
        stage: getStage(),
      };
    },
  };
}
