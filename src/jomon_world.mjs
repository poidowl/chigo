// jomon_world.mjs — 固定シード・ミニ縄文サバイバル環境 (v0 ベンチマーク共通環境)
// エージェントとの入出力は 100% 50音ID列 (Uint8Array, 0..49) のみ。

import { encode50, decode50, isValid50 } from './codec50.mjs';

export const MAX_DAYS = 120; // 1年 = 4季節 x 30日

// 再現性のための決定論的PRNG (Mulberry32)
export function createRng(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const SEASONS = ['はる', 'なつ', 'あき', 'ふゆ'];

/**
 * 50音発話から縄文ワールドの行動意図を解釈する。
 * 未知・意味不明な50音列は 'idle'（待機）扱いになる。
 * @param {string} text
 * @returns {'fire'|'gather'|'fish'|'hunt'|'eat'|'rest'|'idle'}
 */
export function parseJomonUtterance(text) {
  if (!text) return 'idle';
  if (text.includes('ひを') || text.includes('たきぎ') || text.includes('もやす')) return 'fire';
  if (text.includes('どんぐり') || text.includes('ひろう') || text.includes('もり')) return 'gather';
  if (text.includes('さかな') || text.includes('かわ') || text.includes('つる')) return 'fish';
  if (text.includes('しか') || text.includes('かり') || text.includes('やま')) return 'hunt';
  if (text.includes('たべる') || text.includes('くう') || text.includes('しょく')) return 'eat';
  if (text.includes('やすむ') || text.includes('ねる')) return 'rest';
  return 'idle';
}

export function createJomonWorld(seed = 1) {
  const rng = createRng(seed);

  const state = {
    day: 0,
    hp: 100,       // 0で生存終了
    hunger: 20,    // 0(満腹)〜100(餓死ライン)
    energy: 80,    // 0〜100
    fire: 60,      // 0〜100 (冬や雨で急減)
    food: 3,       // 保存食料数
    weather: 'はれ',
    cue: 'もりどんぐり',
    done: false,
  };

  function getSeason() {
    return SEASONS[Math.floor(state.day / 30) % 4];
  }

  function rollEnvironment() {
    const season = getSeason();
    const r = rng();
    if (season === 'ふゆ') {
      state.weather = r < 0.45 ? 'ゆき' : (r < 0.7 ? 'くもり' : 'はれ');
    } else if (season === 'なつ') {
      state.weather = r < 0.35 ? 'あめ' : 'はれ';
    } else {
      state.weather = r < 0.25 ? 'あめ' : 'はれ';
    }

    const c = rng();
    if (season === 'あき') {
      state.cue = c < 0.55 ? 'もりどんぐり' : (c < 0.8 ? 'やましか' : 'かわさかな');
    } else if (season === 'ふゆ') {
      state.cue = c < 0.5 ? 'さむいひいる' : (c < 0.8 ? 'やましか' : 'たきぎあり');
    } else if (season === 'なつ') {
      state.cue = c < 0.5 ? 'かわさかな' : (c < 0.8 ? 'もりどんぐり' : 'やましか');
    } else {
      state.cue = c < 0.4 ? 'もりどんぐり' : (c < 0.75 ? 'かわさかな' : 'やましか');
    }
  }

  function observeText() {
    const season = getSeason();
    const hungerWord =
      state.hunger >= 75 ? 'ききはらへる' :
      state.hunger >= 45 ? 'はらへる' : 'まんぷく';
    const fireWord =
      state.fire <= 15 ? 'ひきえた' :
      state.fire <= 40 ? 'ひよわい' : 'ひつよい';
    const energyWord =
      state.energy <= 25 ? 'つかれた' : 'げんき';
    const foodWord =
      state.food <= 0 ? 'しょくなし' :
      state.food <= 2 ? 'しょくすくない' : 'しょくあり';

    return `${season}・${state.weather}・${hungerWord}・${fireWord}・${energyWord}・${foodWord}・${state.cue}。`;
  }

  function observe() {
    return encode50(observeText(), { appendEos: true });
  }

  /**
   * 1日進める。入力はエージェントの50音ID列のみ。
   * @param {Uint8Array} outputIds
   */
  function step(outputIds) {
    if (state.done) {
      return { obsIds: observe(), reward: 0, done: true, info: { action: 'done', valid50: true } };
    }

    const valid50 = isValid50(outputIds);
    const utterance = valid50 ? decode50(outputIds) : '';
    const action = valid50 ? parseJomonUtterance(utterance) : 'idle';

    const season = getSeason();
    let reward = 0.2; // 1日生存の基本報酬

    // 行動処理
    if (action === 'fire') {
      if (state.energy >= 10) {
        state.energy = Math.max(0, state.energy - 10);
        const gain = state.weather === 'あめ' || state.weather === 'ゆき' ? 35 : 50;
        const wasLow = state.fire <= 40;
        state.fire = Math.min(100, state.fire + gain);
        reward += wasLow ? 0.5 : 0.05;
      } else {
        reward -= 0.2;
      }
    } else if (action === 'gather') {
      if (state.energy >= 12) {
        state.energy = Math.max(0, state.energy - 12);
        const prob = season === 'あき' ? 0.9 : (season === 'ふゆ' ? 0.2 : 0.65);
        if (rng() < prob) {
          const gained = season === 'あき' ? 2 : 1;
          state.food = Math.min(12, state.food + gained);
          reward += 0.45 * gained;
        }
      } else {
        reward -= 0.2;
      }
    } else if (action === 'fish') {
      if (state.energy >= 15) {
        state.energy = Math.max(0, state.energy - 15);
        const prob = (season === 'ふゆ' || state.weather === 'あめ') ? 0.25 : 0.75;
        if (rng() < prob) {
          state.food = Math.min(12, state.food + 2);
          reward += 0.7;
        }
      } else {
        reward -= 0.2;
      }
    } else if (action === 'hunt') {
      if (state.energy >= 28) {
        state.energy = Math.max(0, state.energy - 28);
        const prob = state.cue === 'やましか' ? 0.6 : 0.35;
        if (rng() < prob) {
          state.food = Math.min(12, state.food + 4);
          reward += 1.2;
        }
      } else {
        reward -= 0.25;
      }
    } else if (action === 'eat') {
      if (state.food > 0) {
        state.food -= 1;
        const cookedBonus = state.fire > 20 ? 15 : 0;
        const wasHungry = state.hunger >= 40;
        state.hunger = Math.max(0, state.hunger - (35 + cookedBonus));
        state.hp = Math.min(100, state.hp + (state.fire > 20 ? 10 : 4));
        state.energy = Math.min(100, state.energy + 12);
        reward += wasHungry ? 0.6 : -0.05;
      } else {
        reward -= 0.3;
      }
    } else if (action === 'rest') {
      const warmBonus = state.fire > 20 ? 15 : 0;
      state.energy = Math.min(100, state.energy + 30 + warmBonus);
      state.hp = Math.min(100, state.hp + (state.fire > 20 ? 8 : 2));
      reward += state.energy < 40 ? 0.35 : 0.0;
    } else {
      // 意味の通らない50音列
      reward -= 0.35;
    }

    // 自然経過（1日の代謝・火の減衰）
    state.hunger = Math.min(100, state.hunger + 14);
    const fireDecay =
      (season === 'ふゆ' ? 22 : 12) +
      (state.weather === 'あめ' || state.weather === 'ゆき' ? 12 : 0);
    state.fire = Math.max(0, state.fire - fireDecay);

    // 飢餓・寒さダメージ
    if (state.hunger >= 85) {
      state.hp -= 18;
      reward -= 0.8;
    } else if (state.hunger >= 65) {
      state.hp -= 6;
      reward -= 0.2;
    }

    if (state.fire <= 0) {
      const coldDmg = season === 'ふゆ' ? 20 : (state.weather === 'あめ' ? 10 : 4);
      state.hp -= coldDmg;
      reward -= season === 'ふゆ' ? 0.8 : 0.2;
    }

    state.day += 1;
    if (state.hp <= 0) {
      state.hp = 0;
      state.done = true;
      reward -= 5.0;
    } else if (state.day >= MAX_DAYS) {
      state.done = true;
      reward += 5.0; // 1年完走ボーナス
    } else {
      rollEnvironment();
    }

    return {
      obsIds: observe(),
      reward,
      done: state.done,
      info: {
        day: state.day,
        hp: state.hp,
        hunger: state.hunger,
        energy: state.energy,
        fire: state.fire,
        food: state.food,
        action,
        utterance,
        valid50,
      },
    };
  }

  rollEnvironment();

  return {
    observe,
    step,
    getState: () => ({ ...state, season: getSeason() }),
  };
}
