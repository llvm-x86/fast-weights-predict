'use strict';
// results.js — compute the main experiment tables (mean ± std) and write results.md
const b = require('./bench.js');
const fs = require('fs');

const PREY = ['const-vel', 'circling', 'ou-turn', 'ou-vel', 'jump', 'flee'];
const PREDICTORS = ['velocity-lead', 'accel-lead', 'kalman-lead', 'bdh'];
const POLICIES = ['pure-pursuit', 'mpc', 'linear-q', 'dqn', 'ppo'];
const HORIZON = 24000;

function stats(arr) {
  const n = arr.length;
  const mean = arr.reduce((a, c) => a + c, 0) / n;
  const variance = arr.reduce((a, c) => a + (c - mean) ** 2, 0) / (n - 1);
  return { mean, sd: Math.sqrt(variance) };
}

let out = [];
out.push('# Dragon-Hatchling pursuit benchmark — results');
out.push('');
out.push(`Environment: toroidal ${b.W}×${b.H}, dt=${b.DT}s, catch radius ${b.CATCH_RADIUS}px, chaser max speed 175 px/s (constant), prey speed 165 px/s.`);
out.push('Protocol: reset-on-catch (each catch teleports prey to a random far location, so every catch is a fresh interception from distance).');
out.push('Metric: catches per episode (mean ± sd over seeds).');
out.push('');

// ---------- predictors (lead-pursuit) ----------
out.push('## Lead-pursuit predictors (catches, mean ± sd, reset-on-catch)');
out.push('');
const pSeeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
out.push('| prey | velocity-lead | accel-lead | kalman-lead | bdh (world model) |');
out.push('|---|---|---|---|---|');
for (const pt of PREY) {
  const row = [`| ${pt}`];
  for (const pr of PREDICTORS) {
    const cats = pSeeds.map(s => b.runEpisode(pt, pr, s, HORIZON, { resetOnCatch: true }).catches);
    const { mean, sd } = stats(cats);
    row.push(` ${mean.toFixed(0)} ± ${sd.toFixed(0)}`);
  }
  out.push(row.join(' |') + ' |');
}
out.push('');

// ---------- model-free policies ----------
out.push('## Model-free policies (catches, mean ± sd, reset-on-catch)');
out.push('');
const mSeeds = [1, 2, 3];
const mHorizon = 20000;
out.push('| prey | pure-pursuit (reflex) | MPC (1st-order) | linear-Q (deadly triad) | DQN | PPO |');
out.push('|---|---|---|---|---|---|');
for (const pt of PREY) {
  const row = [`| ${pt}`];
  for (const pl of POLICIES) {
    const cats = mSeeds.map(s => b.runPolicyEpisode(pt, pl, s, mHorizon, { resetOnCatch: true }).catches);
    const { mean, sd } = stats(cats);
    row.push(` ${mean.toFixed(0)} ± ${sd.toFixed(0)}`);
  }
  out.push(row.join(' |') + ' |');
}
out.push('');

fs.writeFileSync('results.md', out.join('\n'));
console.log(out.join('\n'));
console.log('--- written to results.md ---');
