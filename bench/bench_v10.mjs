// bench_v10.mjs — 純ニューラル「かなで話す人」ちご v10 総合検証
// 実行: node bench/bench_v10.mjs

import { createKanaHitoV10, raiseKanaHitoV10 } from '../src/chigo_hito_v10.mjs';

const hito = createKanaHitoV10({ hiddenSize: 288, seed: 108 });
raiseKanaHitoV10(hito, { epochs: 72, seed: 2026 });

// === 部門A: 外部mood注入なしの自律かな会話 ===
console.log('=== 部門A: 外部moodヒントなしの自律「かな会話」 ===');
const CONV_TESTS = [
  { in: 'おはよう',   expect: ['おはよう'] },
  { in: 'こんにちは', expect: ['こんにちは'] },
  { in: 'なまえ',     expect: ['ちご'] },
  { in: 'だれ',       expect: ['ちご', 'ひと'] },
  { in: 'ありがとう', expect: ['うれしい'] },
  { in: 'すき',       expect: ['うれしい', 'すき'] },
  { in: 'そら',       expect: ['ひろい'] },
  { in: 'はな',       expect: ['きれい'] },
  { in: 'とり',       expect: ['そら', 'とぶ'] },
  { in: 'くま',       expect: ['にげ'] },
  { in: 'さむ',       expect: ['ひ', 'たす'] },
  { in: 'はら',       expect: ['み', 'くう'] },
  { in: 'おやすみ',   expect: ['ねる'] },
];

let convHits = 0;
for (const t of CONV_TESTS) {
  hito.resetState(true);
  const res = hito.probeText(t.in); // mood を一切渡さない！
  const ok = t.expect.some((kw) => res.reply.includes(kw));
  if (ok) convHits += 1;
  console.log(
    `  ${t.in.padEnd(7, '　')} [驚:${res.surprise.toFixed(2)}] => ${res.reply.padEnd(10, '　')} ${ok ? '✓' : '✗'}`
  );
}

// === 部門B: ターン跨ぎ自律こころ分岐（v9では外部mood必須だった課題） ===
console.log('\n=== 部門B: ターン跨ぎ自律「こころ」分岐（外部moodなし） ===');
let branchHits = 0;

// シナリオ1: 1ターン目「さむ」→ 2ターン目「あめ」=> ひ・きえ。
hito.resetState(true);
const s1t1 = hito.talkText('さむ');
const s1t2 = hito.talkText('あめ');
const s1Ok = s1t2.reply.includes('ひ') || s1t2.reply.includes('きえ');
if (s1Ok) branchHits += 1;
console.log(`  1) 「さむ」=> ${s1t1.reply} (さむ心:${s1t1.kokoro[0].toFixed(2)}) → 「あめ」=> ${s1t2.reply} ${s1Ok ? '✓' : '✗'}`);

// シナリオ2: 1ターン目「はら」→ 2ターン目「あめ」=> みず・のむ。
hito.resetState(true);
const s2t1 = hito.talkText('はら');
const s2t2 = hito.talkText('あめ');
const s2Ok = s2t2.reply.includes('みず') || s2t2.reply.includes('のむ');
if (s2Ok) branchHits += 1;
console.log(`  2) 「はら」=> ${s2t1.reply} (はら心:${s2t1.kokoro[1].toFixed(2)}) → 「あめ」=> ${s2t2.reply} ${s2Ok ? '✓' : '✗'}`);

// シナリオ3: 1ターン目「さむ」→ 2ターン目「はら」→ 3ターン目「あめ」=> 心が切り替わり みず・のむ。
hito.resetState(true);
hito.talkText('さむ');
const s3t2 = hito.talkText('はら');
const s3t3 = hito.talkText('あめ');
const s3Ok = s3t3.reply.includes('みず') || s3t3.reply.includes('のむ');
if (s3Ok) branchHits += 1;
console.log(`  3) 「さむ」→「はら」(はら心:${s3t2.kokoro[1].toFixed(2)}) →「あめ」=> ${s3t3.reply} ${s3Ok ? '✓' : '✗'}`);

// === 部門C: オンライン未知語習得＋訂正学習 ===
console.log('\n=== 部門C: 昼のオンライン未知語習得＆訂正学習 ===');
hito.resetState(true);
const NOVEL = [
  { unk: 'ほし', teach: 'うえ・ひ。', expect: ['うえ', 'ひ'] },
  { unk: 'ゆめ', teach: 'よ・みる。', expect: ['よ', 'みる'] },
  { unk: 'ねこ', teach: 'かわいい。', expect: ['かわいい'] },
];

let novelAskHits = 0;
let novelLearnHits = 0;
for (const n of NOVEL) {
  hito.resetState(true);
  const r1 = hito.talkText(n.unk);
  const asked = r1.reply.includes('なに');
  if (asked) novelAskHits += 1;
  const r2 = hito.talkText(n.teach);
  const r3 = hito.probeText(n.unk);
  const learnedOk = n.expect.some((kw) => r3.reply.includes(kw));
  if (learnedOk) novelLearnHits += 1;
  console.log(
    `  未知語「${n.unk}」: 1)初見[驚:${r1.surprise.toFixed(2)}]=>${r1.reply}  2)教え「${n.teach}」=>${r2.reply}  3)確認=>${r3.reply} ${asked && learnedOk ? '✓' : '✗'}`
  );
}

// 訂正学習テスト: 「ねこ」の返事を「にやあ。」に訂正（50音仕様で小文字ゃは「や」へ正規化）
hito.resetState(true);
hito.talkText('ねこ');
const corrected = hito.correctLastText('にゃあ。');
const correctOk = corrected && corrected.reply.includes('にやあ');
console.log(`  訂正学習「ねこ」=> 「にやあ。」確認: ${corrected?.reply} ${correctOk ? '✓' : '✗'}`);

// === 部門D: 睡眠リプレイ定着 (W_fast=0) ===
console.log('\n=== 部門D: 夜の睡眠リプレイ定着 (W_fast = 0 へ完全消去) ===');
hito.sleepConsolidate(28);
const fastAfterSleep = hito.getFastNorm();

let sleepOldHits = 0;
for (const t of CONV_TESTS) {
  hito.resetState(true);
  const res = hito.probeText(t.in);
  if (t.expect.some((kw) => res.reply.includes(kw))) sleepOldHits += 1;
}

const SLEEP_NEW_CHECKS = [
  { in: 'ほし', expect: ['うえ', 'ひ'] },
  { in: 'ゆめ', expect: ['よ', 'みる'] },
  { in: 'ねこ', expect: ['にやあ'] },
];
let sleepNewHits = 0;
for (const c of SLEEP_NEW_CHECKS) {
  hito.resetState(true);
  const res = hito.probeText(c.in);
  const ok = c.expect.some((kw) => res.reply.includes(kw));
  if (ok) sleepNewHits += 1;
  console.log(`  定着確認「${c.in}」 [驚:${res.surprise.toFixed(2)}] => ${res.reply} ${ok ? '✓' : '✗'}`);
}
hito.resetState(true);
const stillUnk = hito.probeText('にじ');
const stillUnkOk = stillUnk.reply.includes('なに');
console.log(`  新未知語「にじ」 [驚:${stillUnk.surprise.toFixed(2)}] => ${stillUnk.reply} ${stillUnkOk ? '✓' : '✗'}`);

// === 部門E: 脳の保存・復元 (serialize / deserialize) ===
const snapshot = JSON.parse(JSON.stringify(hito.serialize()));
const restored = createKanaHitoV10({ hiddenSize: snapshot.hiddenSize, seed: snapshot.seed });
restored.deserialize(snapshot);
restored.resetState(true);
const rCheck1 = restored.probeText('おはよう').reply;
const rCheck2 = restored.probeText('ねこ').reply;
const saveOk = rCheck1.includes('おはよう') && rCheck2.includes('にやあ');

console.log('\n=== 純ニューラル「かなで話す人」ちご v10 スコアまとめ ===');
console.table([
  {
    部門: 'A: 自律かな会話(moodなし)',
    結果: `${((convHits / CONV_TESTS.length) * 100).toFixed(1)}% (${convHits}/${CONV_TESTS.length})`,
  },
  {
    部門: 'B: ターン跨ぎ自律こころ分岐',
    結果: `${((branchHits / 3) * 100).toFixed(1)}% (${branchHits}/3)`,
  },
  {
    部門: 'C: オンライン習得＋訂正',
    結果: `聞き返し${novelAskHits}/3, 習得${novelLearnHits}/3, 訂正${correctOk ? 'OK' : 'NG'}`,
  },
  {
    部門: 'D: 睡眠定着後 (W_fast=0)',
    結果: `既存${sleepOldHits}/${CONV_TESTS.length}, 新規${sleepNewHits}/3, W_fast=${fastAfterSleep}, 未知語=${stillUnkOk ? 'OK' : 'NG'}`,
  },
  {
    部門: 'E: 脳の保存・復元',
    結果: saveOk ? '100% (OK)' : 'NG',
  },
]);
