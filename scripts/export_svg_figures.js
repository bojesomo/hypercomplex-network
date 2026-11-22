// Export the inline SVG figures of docs/pdf_src/hyperavnet.html as standalone vector PDFs for the paper.
// usage: NODE_PATH=$(npm root -g) node scripts/export_svg_figures.js docs/pdf_src/hyperavnet.html paper/figures
const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const [src, outDir] = process.argv.slice(2);
  const names = ['arch_overview', 'hamilton', 'hyper_ir', 'hcf_block', 'fusion_schedule', 'distillation'];
  const b = await chromium.launch();
  const p = await b.newPage();
  await p.goto('file://' + path.resolve(src));
  const svgs = await p.$$eval('figure svg', els => els.map(e => e.outerHTML));
  for (let i = 0; i < Math.min(svgs.length, names.length); i++) {
    const q = await b.newPage();
    await q.setContent(`<html><head><style>html,body{margin:0;background:#fff}svg{display:block;width:1000px;height:auto}
      svg text{font-family:Inter,'Liberation Sans',sans-serif}</style></head><body>${svgs[i]}</body></html>`);
    const box = await q.$eval('svg', e => { const r = e.getBoundingClientRect(); return [Math.ceil(r.width), Math.ceil(r.height)]; });
    await q.pdf({ path: path.join(outDir, names[i] + '.pdf'), width: box[0] + 'px', height: (box[1] + 2) + 'px',
                  printBackground: true, pageRanges: '1' });
    await q.close();
    console.log(names[i], box);
  }
  await b.close();
})();
