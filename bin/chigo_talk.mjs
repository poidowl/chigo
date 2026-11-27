#!/usr/bin/env node
// chigo_talk.mjs — 純ニューラル「かなで話す人」ちご v10 対話CLI
// 実行: npm run talk  または  node bin/chigo_talk.mjs [brain.json]

import fs from 'node:fs';
import readline from 'node:readline';
import { createKanaHitoV10, raiseKanaHitoV10, EMO_NAMES } from '../src/chigo_hito_v10.mjs';

const brainPath = process.argv[2] || null;
const hito = createKanaHitoV10({ hiddenSize: 288, seed: 108 });

if (brainPath && fs.existsSync(brainPath)) {
  const data = JSON.parse(fs.readFileSync(brainPath, 'utf8'));
  hito.deserialize(data);
  console.log(`[のう] ${brainPath} をよみこみました（海馬:${hito.getHippoSize()}件）`);
} else {
  process.stdout.write('[そだてちゅう] ちご v10 を 50おん だけで そだてています...');
  raiseKanaHitoV10(hito, { epochs: 72, seed: 2026 });
  console.log(' できた！');
}

function formatKokoro(k) {
  return EMO_NAMES.map((name, i) => `${name}:${k[i].toFixed(2)}`).join(' ');
}

console.log('');
console.log('=== ちご v10（かなで話すひと・純ニューラル50音） ===');
console.log('ひらがな で はなしかけてね。しらないことば は「なに。」とききかえすよ。');
console.log('コマンド: /sleep (ねむってていちゃく)  /fix <へんじ> (ちょくぜんをなおす)');
console.log('          /teach <ことば> => <へんじ>  /save [file]  /mood  /reset  /quit');
console.log('');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  prompt: 'あなた> ',
});

rl.prompt();

rl.on('line', (rawLine) => {
  const line = rawLine.trim();
  if (!line) {
    rl.prompt();
    return;
  }

  if (line === '/quit' || line === '/exit') {
    console.log('ちご  > またね。');
    rl.close();
    return;
  }

  if (line === '/mood') {
    const s = hito.getStats();
    console.log(`[こころ] ${formatKokoro(s.kokoro)} | W_fast:${s.fastNorm} | 海馬:${s.hippoSize} | 睡眠:${s.sleepCount}`);
    rl.prompt();
    return;
  }

  if (line === '/reset') {
    hito.resetState(true);
    console.log('[りせっと] かいわの ながれ と こころ を しずめました。');
    rl.prompt();
    return;
  }

  if (line === '/sleep' || line === 'ねむる') {
    process.stdout.write('[ねむり] ゆめ リプレイ ていちゃくちゅう...');
    hito.sleepConsolidate(28);
    console.log(` おきた！（W_fast:${hito.getFastNorm()} / 海馬:${hito.getHippoSize()}）`);
    rl.prompt();
    return;
  }

  if (line.startsWith('/fix ')) {
    const better = line.slice(5).trim();
    const res = hito.correctLastText(better);
    if (!res) {
      console.log('[おしらせ] まだ ちょくぜんの はつわ が ないよ。');
    } else {
      console.log(`[なおした] => ${res.reply} (W_fast:${hito.getFastNorm()})`);
    }
    rl.prompt();
    return;
  }

  if (line.startsWith('/teach ')) {
    const body = line.slice(7);
    const parts = body.split('=>');
    if (parts.length !== 2) {
      console.log('つかいかた: /teach ほし => うえ・ひ。');
    } else {
      const res = hito.teachText(parts[0].trim(), parts[1].trim());
      console.log(`[おぼえた] 「${parts[0].trim()}」=> ${res.reply} (W_fast:${hito.getFastNorm()})`);
    }
    rl.prompt();
    return;
  }

  if (line.startsWith('/save')) {
    const outPath = line.slice(5).trim() || brainPath || 'chigo_brain_v10.json';
    fs.writeFileSync(outPath, JSON.stringify(hito.serialize()), 'utf8');
    console.log(`[ほぞん] ${outPath} へ のう を ほぞんしました。`);
    rl.prompt();
    return;
  }

  const res = hito.talkText(line);
  const tag = res.learned ? ' [おぼえた!]' : res.surprise > 0.65 ? ' [しらないことば→つぎに おしえてね]' : '';
  console.log(`ちご  > ${res.reply}${tag}`);
  console.log(`        [驚:${res.surprise.toFixed(2)} | ${formatKokoro(res.kokoro)}]`);
  rl.prompt();
});
