// codec50.mjs — 固定50音トークン仕様 (v0 凍結仕様)
// ID 0..49 のちょうど50トークンだけで全入出力を表現する。
// 0: _ (PAD/無音), 1: ・ (区切り), 2: ゛ (濁点), 3: 。 (EOS/発話終了)
// 4..49: 清音46文字 (あ〜ん)

export const VOCAB_SIZE = 50;

export const TOKEN_PAD = 0;
export const TOKEN_SEP = 1;
export const TOKEN_DAKUTEN = 2;
export const TOKEN_EOS = 3;

export const VOCAB = [
  '_', '・', '゛', '。',
  'あ', 'い', 'う', 'え', 'お',
  'か', 'き', 'く', 'け', 'こ',
  'さ', 'し', 'す', 'せ', 'そ',
  'た', 'ち', 'つ', 'て', 'と',
  'な', 'に', 'ぬ', 'ね', 'の',
  'は', 'ひ', 'ふ', 'へ', 'ほ',
  'ま', 'み', 'む', 'め', 'も',
  'や', 'ゆ', 'よ',
  'ら', 'り', 'る', 'れ', 'ろ',
  'わ', 'を', 'ん',
];

if (VOCAB.length !== VOCAB_SIZE) {
  throw new Error(`VOCAB size mismatch: ${VOCAB.length}`);
}

const CHAR_TO_ID = new Map(VOCAB.map((ch, idx) => [ch, idx]));

// 濁音・半濁音 → [清音, 濁点] に分解（半濁音も50トークン維持のため濁点に正規化）
const DAKUTEN_DECOMPOSE = {
  'が': 'か', 'ぎ': 'き', 'ぐ': 'く', 'げ': 'け', 'ご': 'こ',
  'ざ': 'さ', 'じ': 'し', 'ず': 'す', 'ぜ': 'せ', 'ぞ': 'そ',
  'だ': 'た', 'ぢ': 'ち', 'づ': 'つ', 'で': 'て', 'ど': 'と',
  'ば': 'は', 'び': 'ひ', 'ぶ': 'ふ', 'べ': 'へ', 'ぼ': 'ほ',
  'ぱ': 'は', 'ぴ': 'ひ', 'ぷ': 'ふ', 'ぺ': 'へ', 'ぽ': 'ほ',
  'ゔ': 'う',
};

const DAKUTEN_COMPOSE = {
  'か': 'が', 'き': 'ぎ', 'く': 'ぐ', 'け': 'げ', 'こ': 'ご',
  'さ': 'ざ', 'し': 'じ', 'す': 'ず', 'せ': 'ぜ', 'そ': 'ぞ',
  'た': 'だ', 'ち': 'ぢ', 'つ': 'づ', 'て': 'で', 'と': 'ど',
  'は': 'ば', 'ひ': 'び', 'ふ': 'ぶ', 'へ': 'べ', 'ほ': 'ぼ',
  'う': 'ゔ',
};

const SMALL_TO_BASE = {
  'ぁ': 'あ', 'ぃ': 'い', 'ぅ': 'う', 'ぇ': 'え', 'ぉ': 'お',
  'っ': 'つ', 'ゃ': 'や', 'ゅ': 'ゆ', 'ょ': 'よ', 'ゎ': 'わ',
};

/**
 * 任意の日本語文字列を厳密な50音ID列 (Uint8Array, 値域 0..49) に変換する。
 * @param {string} text
 * @param {{ appendEos?: boolean }} [opts]
 * @returns {Uint8Array}
 */
export function encode50(text, opts = {}) {
  const { appendEos = false } = opts;
  const out = [];

  for (const rawCh of String(text)) {
    // カタカナをひらがなへ正規化
    const code = rawCh.codePointAt(0);
    let ch = (code >= 0x30A1 && code <= 0x30F6)
      ? String.fromCodePoint(code - 0x60)
      : rawCh;

    if (ch === ' ' || ch === '\u3000' || ch === '、' || ch === ',') {
      ch = '・';
    } else if (ch === 'ー') {
      // 長音は直前の母音を引き継ぐ代わりに区切りへ落とすか省略（50枠固定）
      continue;
    }

    if (SMALL_TO_BASE[ch]) {
      ch = SMALL_TO_BASE[ch];
    }

    if (DAKUTEN_DECOMPOSE[ch]) {
      const baseId = CHAR_TO_ID.get(DAKUTEN_DECOMPOSE[ch]);
      out.push(baseId, TOKEN_DAKUTEN);
      continue;
    }

    const id = CHAR_TO_ID.get(ch);
    if (id !== undefined) {
      out.push(id);
    }
  }

  if (appendEos && (out.length === 0 || out[out.length - 1] !== TOKEN_EOS)) {
    out.push(TOKEN_EOS);
  }

  return Uint8Array.from(out);
}

/**
 * 50音ID列を人間が読めるひらがな文字列にデコードする。
 * @param {Uint8Array | number[]} ids
 * @param {{ raw?: boolean }} [opts]
 * @returns {string}
 */
export function decode50(ids, opts = {}) {
  const { raw = false } = opts;
  if (raw) {
    let s = '';
    for (let i = 0; i < ids.length; i++) {
      s += VOCAB[ids[i]] ?? '?';
    }
    return s;
  }

  let out = '';
  for (let i = 0; i < ids.length; i++) {
    const id = ids[i];
    if (id === TOKEN_PAD) continue;
    const ch = VOCAB[id];
    if (!ch) continue;
    if (id === TOKEN_DAKUTEN && out.length > 0) {
      const prev = out[out.length - 1];
      if (DAKUTEN_COMPOSE[prev]) {
        out = out.slice(0, -1) + DAKUTEN_COMPOSE[prev];
        continue;
      }
    }
    out += ch;
  }
  return out;
}

/**
 * 厳密に 0..49 の範囲に収まっているか検証する。
 * @param {Uint8Array | number[]} ids
 * @returns {boolean}
 */
export function isValid50(ids) {
  if (!ids || typeof ids.length !== 'number') return false;
  for (let i = 0; i < ids.length; i++) {
    const v = ids[i];
    if (!Number.isInteger(v) || v < 0 || v >= VOCAB_SIZE) return false;
  }
  return true;
}
