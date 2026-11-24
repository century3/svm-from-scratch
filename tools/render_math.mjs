// Builds README.md from README.src.md, replacing every LaTeX formula with an SVG image.
// GitHub renders math client-side from text, so page translators rewrite it (max -> 马克思,
// \xi -> \习, "," -> "，"); images are left alone. Edit README.src.md, then run:
//   cd tools && npm install && node render_math.mjs
import { createRequire } from 'node:module';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const { mathjax } = require('mathjax-full/js/mathjax.js');
const { TeX } = require('mathjax-full/js/input/tex.js');
const { SVG } = require('mathjax-full/js/output/svg.js');
const { liteAdaptor } = require('mathjax-full/js/adaptors/liteAdaptor.js');
const { RegisterHTMLHandler } = require('mathjax-full/js/handlers/html.js');
const { AllPackages } = require('mathjax-full/js/input/tex/AllPackages.js');

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const srcFile = path.join(root, 'README.src.md');
const outFile = path.join(root, 'README.md');
const mathDir = path.join(root, 'images', 'math');
const relDir = 'images/math';

const LIGHT = '#1f2328', DARK = '#e6edf3';   // GitHub's default text colours
const PX_PER_EX = 8;                          // GitHub body text is 16px; x-height ~ 0.5em

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const doc = mathjax.document('', {
  InputJax: new TeX({ packages: AllPackages }),
  OutputJax: new SVG({ fontCache: 'none' }),   // self-contained paths, needed inside <img>
});

const num = (s) => parseFloat(s);

function renderSvg(tex, display) {
  const node = doc.convert(tex, { display });
  let svg = adaptor.innerHTML(node);
  if (svg.includes('data-mjx-error') || svg.includes('merror')) {
    throw new Error(`MathJax error in formula: ${tex}`);
  }
  const attr = (name) => (svg.match(new RegExp(`\\s${name}="([^"]*)"`)) || [])[1];
  let hEx = num(attr('height')), wEx = num(attr('width'));
  let [vx, vy, vw, vh] = attr('viewBox').split(/\s+/).map(Number);
  const depthEx = -num((attr('style') || '').match(/vertical-align:\s*([-\d.]+)ex/)?.[1] || 0);

  if (!display) {
    // <img align="middle"> centres the image on the text baseline (Chrome/Firefox: baseline-middle);
    // pad the SVG so that the formula's own baseline sits at the image's vertical centre.
    const unitsPerEx = vh / hEx;
    const pad = hEx - 2 * depthEx;
    if (pad > 0) { vh += pad * unitsPerEx; hEx += pad; }                       // pad bottom
    else if (pad < 0) { vy += pad * unitsPerEx; vh -= pad * unitsPerEx; hEx -= pad; } // pad top
  }
  const w = (wEx * PX_PER_EX).toFixed(2), h = (hEx * PX_PER_EX).toFixed(2);
  svg = svg
    .replace(/\sstyle="[^"]*"/, '')
    .replace(/\swidth="[^"]*"/, ` width="${w}"`)
    .replace(/\sheight="[^"]*"/, ` height="${h}"`)
    .replace(/\sviewBox="[^"]*"/, ` viewBox="${vx} ${vy} ${vw} ${vh}"`)
    .replace(/\s(role|focusable|aria-hidden)="[^"]*"/g, '');
  return svg;
}

const escAttr = (s) => s.replace(/\s+/g, ' ').trim()
  .replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  .replace(/\|/g, '&#124;');

const written = new Set();
function image(tex, display) {
  const id = createHash('sha1').update((display ? 'D:' : 'I:') + tex).digest('hex').slice(0, 12);
  if (!written.has(id)) {
    const svg = renderSvg(tex, display);
    fs.writeFileSync(path.join(mathDir, `${id}.svg`), svg.replaceAll('currentColor', LIGHT));
    fs.writeFileSync(path.join(mathDir, `${id}-dark.svg`), svg.replaceAll('currentColor', DARK));
    written.add(id);
  }
  const img = `<img src="${relDir}/${id}.svg" alt="${escAttr(tex)}"${display ? '' : ' align="middle"'}>`;
  return `<picture><source media="(prefers-color-scheme: dark)" srcset="${relDir}/${id}-dark.svg">${img}</picture>`;
}

fs.rmSync(mathDir, { recursive: true, force: true });
fs.mkdirSync(mathDir, { recursive: true });

const lines = fs.readFileSync(srcFile, 'utf8').replace(/\r\n/g, '\n').split('\n');
const out = ['<!-- 本文件由 tools/render_math.mjs 根据 README.src.md 自动生成，请编辑 README.src.md -->', ''];
let fence = null, mathBuf = [], nInline = 0, nDisplay = 0;
for (const line of lines) {
  const t = line.trimStart();
  if (fence === 'math') {
    if (t.startsWith('```')) {
      out.push('', `<p align="center">${image(mathBuf.join('\n'), true)}</p>`, '');
      fence = null; mathBuf = []; nDisplay++;
    } else mathBuf.push(line);
    continue;
  }
  if (fence === 'code') { out.push(line); if (t.startsWith('```')) fence = null; continue; }
  if (t.startsWith('```math')) { fence = 'math'; continue; }
  if (t.startsWith('```')) { fence = 'code'; out.push(line); continue; }
  out.push(line.replace(/\$`([^`]+)`\$/g, (_, tex) => { nInline++; return image(tex, false); }));
}
if (fence) throw new Error('Unclosed code fence in README.src.md');
fs.writeFileSync(outFile, out.join('\n').replace(/\n{3,}/g, '\n\n'));
console.log(`README.md written: ${nDisplay} display + ${nInline} inline formulas, ${written.size * 2} SVG files`);
