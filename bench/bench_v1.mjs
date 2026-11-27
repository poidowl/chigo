// bench_v1.mjs — 「ヤマトコトバ原始人AGI」発達＆知能ベンチマーク (v1)
// 実行: node bench/bench_v1.mjs
// 評価軸:
//   1. 短語制約 (1〜2文字語のみ・3文字以上ゼロ・50音のみ)
//   2. 因果・行動推論 (雨→火が消える、熊→逃げ、腹へる→食う 等)
//   3. 未知語の問い返し＆即時記憶 (知らない語を聞いて覚え、次の会話で使えるか)
//   4. 発達遷移 (Stage 0 一語文 → Stage 1 二語文 → Stage 2 因果連鎖)

import { decode50, isValid50 } from '../src/codec50.mjs';
import { createJomonBabyV1 } from '../src/jomon_baby_v1.mjs';

function checkShortWordCompliance(replyText) {
  const words = replyText
    .replace(/。/g, '・')
    .split('・')
    .filter(Boolean);
  if (words.length === 0) return false;
  return words.every((w) => w.length >= 1 && w.length <= 2);
}

function runDialogueBench() {
  const baby = createJomonBabyV1({ startStage: 0 });

  // 赤ん坊→幼児→原始人AGIへと育てながらテストする固定シナリオ
  const script = [
    // --- Stage 0 (赤ん坊・一語文の時期) ---
    { in: 'ひ', expectAny: ['ひ', 'あつ', 'もえ'], phase: 'Stage0-一語文' },
    { in: 'みず', expectAny: ['みず', 'のむ'], phase: 'Stage0-一語文' },
    { in: 'くま', expectAny: ['くま', 'こわ', 'にげ'], phase: 'Stage0-一語文' },
    { in: 'あめ', expectAny: ['あめ', 'ひ', 'きえ', 'みず'], phase: 'Stage0-一語文' },

    // --- Stage 1 (幼児・二語文の芽生え) ---
    { in: 'くま・くる', expectAny: ['こわ', 'にげ'], phase: 'Stage1-二語文' },
    { in: 'はら・へる', expectAny: ['くう', 'み'], phase: 'Stage1-二語文' },
    { in: 'さむ', expectAny: ['ひ', 'たす'], phase: 'Stage1-二語文' },
    // 未知語「ほし」を初めて聞く → 問い返すか？
    { in: 'ほし', expectAny: ['なに'], phase: 'Stage1-未知語質問' },
    // 教える：「ほし・うえ・ひ」（星は上の火）
    { in: 'うえ・ひ', expectAny: ['うえ', 'ひ', 'よい'], phase: 'Stage1-記憶' },

    // --- Stage 2 (原始人AGI・二語文の因果連鎖＋記憶活用) ---
    // さっき覚えた「ほし」を推論に使えるか？
    { in: 'ほし', expectAny: ['うえ', 'ひ'], phase: 'Stage2-記憶活用' },
    // 2ホップ因果: あめ → ひ・きえ → ひ・き（木を足す）
    { in: 'あめ・くる', expectAny: ['きえ', 'ひ'], phase: 'Stage2-因果連鎖' },
    { in: 'くま・くる', expectAny: ['にげ'], phase: 'Stage2-危機回避' },
    { in: 'しか・ある', expectAny: ['とる', 'くう', 'うま'], phase: 'Stage2-狩猟判断' },
    // 2つ目の未知語「ゆき」
    { in: 'ゆき', expectAny: ['なに'], phase: 'Stage2-未知語仮説' },
    { in: 'さむ・あめ', expectAny: ['さむ', 'あめ', 'よい'], phase: 'Stage2-記憶' },
    { in: 'ゆき', expectAny: ['さむ', 'あめ'], phase: 'Stage2-記憶活用2' },
  ];

  let shortOk = 0;
  let valid50Ok = 0;
  let logicHit = 0;
  const stagesSeen = new Set();

  console.log('=== 発達対話ログ (v1-jomon-baby / 3文字以上封印) ===');
  for (const turn of script) {
    const res = baby.talk(turn.in);
    const v50 = isValid50(res.outIds);
    const shortPass = checkShortWordCompliance(res.reply);
    const hit = turn.expectAny.some((kw) => res.reply.includes(kw));

    if (v50) valid50Ok += 1;
    if (shortPass) shortOk += 1;
    if (hit) logicHit += 1;
    stagesSeen.add(res.stage);

    console.log(
      `[St${res.stage}][${turn.phase}] ひと: ${turn.in.padEnd(7, '　')} => AGI: ${res.reply} ${hit ? '✓' : '✗'}`
    );
  }

  const total = script.length;
  const shortRate = (shortOk / total) * 100;
  const valid50Rate = (valid50Ok / total) * 100;
  const logicRate = (logicHit / total) * 100;
  const devScore = (stagesSeen.size / 3) * 100;
  const totalScore =
    logicRate * 0.5 + shortRate * 0.2 + valid50Rate * 0.1 + devScore * 0.2;

  console.log('\n=== v1 ベンチマーク結果 ===');
  console.table([
    {
      name: baby.name,
      shortWordRate: `${shortRate.toFixed(1)}%`,
      valid50Rate: `${valid50Rate.toFixed(1)}%`,
      reasoningAndMemory: `${logicRate.toFixed(1)}%`,
      stagesReached: `${stagesSeen.size}/3`,
      vocabLearned: baby.getVocabSize(),
      score: Number(totalScore.toFixed(2)),
    },
  ]);
}

runDialogueBench();