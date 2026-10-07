// "Pick one fit vs average all fits" on bs_smile with one example (k = 1).
// One option price can't pin down three smile parameters, so many smiles fit it
// exactly. Calibration (least squares) returns one of them; the Bayes-optimal
// prediction averages the query price over all of them, weighted by the prior.
//
// Markup:  <div class="posterior"></div>   (needs bs.js loaded first)

(function () {
  const { callOverSpot, features } = window.BS;
  const STD = 0.10631866753101349; // BlackScholesSmile.moments() std, from src/tasks.py
  const TOL = 0.002; // "fits the example": smile vol at the example strike within ±0.002
  const DRAWS = 10000;

  const U = (a, b) => a + (b - a) * Math.random();
  const randn = () => Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
  const prior = () => [U(0.15, 0.4), U(-0.4, 0), U(0, 1)]; // sigma0, skew, curvature
  const vol = (p, m) => Math.max(0.05, p[0] + p[1] * m + p[2] * m * m);
  const pct = (p) => (100 * p).toFixed(2) + "%";
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;

  // One stock: hidden smile, one example, one query; all smiles that fit the example
  function stock(spread) {
    const truth = prior();
    const ex = features(randn(), randn());
    let q;
    do q = features(randn(), randn()); while (spread && Math.abs(q.m - ex.m) < 0.12);
    const target = vol(truth, ex.m);
    const fits = [];
    let priorSum = 0;
    for (let i = 0; i < DRAWS; i++) {
      const p = prior();
      const price = callOverSpot(q.m, q.T, vol(p, q.m));
      priorSum += price;
      if (Math.abs(vol(p, ex.m) - target) < TOL) fits.push({ p, price });
    }
    if (fits.length < 5) return stock(spread);
    return {
      truth, ex, q, fits,
      truePrice: callOverSpot(q.m, q.T, vol(truth, q.m)),
      pick: fits[Math.floor(Math.random() * fits.length)],
      avg: mean(fits.map((f) => f.price)),
      priorMean: priorSum / DRAWS,
    };
  }

  // ---- drawing (inline SVG, colours from course.css) ----
  function smilePanel(s) {
    const W = 300, H = 190, P = 30;
    const x = (m) => P + ((m + 0.5) / 1.0) * (W - P - 8);
    const y = (v) => H - P - (v / 0.8) * (H - P - 8);
    const path = (p) => {
      let d = "";
      for (let i = 0; i <= 40; i++) {
        const m = -0.5 + i / 40;
        d += (i ? "L" : "M") + x(m).toFixed(1) + "," + y(vol(p, m)).toFixed(1);
      }
      return d;
    };
    const shown = s.fits.slice(0, 40);
    return `<svg viewBox="0 0 ${W} ${H}" class="post-svg" role="img" aria-label="Smiles that fit the example">
      <line x1="${P}" y1="${H - P}" x2="${W - 8}" y2="${H - P}" stroke="var(--rule)"/>
      <line x1="${P}" y1="8" x2="${P}" y2="${H - P}" stroke="var(--rule)"/>
      ${shown.map((f) => `<path d="${path(f.p)}" fill="none" stroke="var(--accent)" stroke-opacity="0.18"/>`).join("")}
      <path d="${path(s.pick.p)}" fill="none" stroke="var(--bad)" stroke-width="1.6" stroke-dasharray="4 3"/>
      <path d="${path(s.truth)}" fill="none" stroke="var(--ink)" stroke-width="2"/>
      <line x1="${x(s.q.m)}" y1="8" x2="${x(s.q.m)}" y2="${H - P}" stroke="var(--muted)" stroke-dasharray="2 3"/>
      <circle cx="${x(s.ex.m)}" cy="${y(vol(s.truth, s.ex.m))}" r="4.5" fill="var(--accent)"/>
      <text x="${x(s.ex.m) + 7}" y="${y(vol(s.truth, s.ex.m)) - 7}" font-size="10" fill="var(--accent)">example</text>
      <text x="${x(s.q.m) + 4}" y="18" font-size="10" fill="var(--muted)">query strike</text>
      <text x="${W - 8}" y="${H - 8}" font-size="10" text-anchor="end" fill="var(--muted)">log-moneyness m</text>
      <text x="${P - 4}" y="14" font-size="10" text-anchor="end" fill="var(--muted)">0.8</text>
      <text x="${P - 4}" y="${H - P}" font-size="10" text-anchor="end" fill="var(--muted)">0</text>
      <text x="${P + 4}" y="${H - P - 4}" font-size="10" fill="var(--muted)">σ(m)</text>
    </svg>`;
  }

  function pricePanel(s) {
    const W = 300, H = 120, P = 12;
    const prices = s.fits.map((f) => f.price).concat([s.truePrice]);
    let lo = Math.min(...prices), hi = Math.max(...prices);
    const pad = (hi - lo) * 0.08 + 1e-4;
    lo -= pad; hi += pad;
    const x = (p) => P + ((p - lo) / (hi - lo)) * (W - 2 * P);
    const mark = (p, colour, label, row) =>
      `<line x1="${x(p)}" y1="12" x2="${x(p)}" y2="78" stroke="${colour}" stroke-width="2"/>
       <text x="${x(p)}" y="${92 + 12 * row}" font-size="10" text-anchor="middle" fill="${colour}">${label}</text>`;
    return `<svg viewBox="0 0 ${W} ${H}" class="post-svg" role="img" aria-label="Query prices of all fitting smiles">
      ${s.fits.map((f) => `<circle cx="${x(f.price).toFixed(1)}" cy="${(20 + Math.random() * 50).toFixed(1)}" r="2.2" fill="var(--accent)" fill-opacity="0.35"/>`).join("")}
      ${mark(s.truePrice, "var(--ink)", "true", 0)}
      ${mark(s.avg, "var(--good)", "average", 1)}
      ${mark(s.pick.price, "var(--bad)", "one fit", 2)}
    </svg>`;
  }

  function build(el) {
    el.innerHTML = `
      <div class="post-panels">
        <figure><div class="post-a"></div><figcaption>Faint: smiles that price the example exactly. Black: the true smile. Dashed red: the one a least-squares fit might return.</figcaption></figure>
        <figure><div class="post-b"></div><figcaption>Each dot: the query price under one fitting smile.</figcaption></figure>
      </div>
      <p class="post-read"></p>
      <p class="calib-buttons">
        <button type="button" class="post-new">New stock</button>
        <button type="button" class="post-run lock">Score 1,000 stocks</button>
      </p>
      <div class="post-score"></div>`;
    const $ = (sel) => el.querySelector(sel);

    function show() {
      const s = stock(true);
      $(".post-a").innerHTML = smilePanel(s);
      $(".post-b").innerHTML = pricePanel(s);
      $(".post-read").innerHTML =
        `${s.fits.length} of ${DRAWS.toLocaleString()} prior smiles fit the example. Query price: true <b>${pct(s.truePrice)}</b>, ` +
        `average of fits <b style="color:var(--good)">${pct(s.avg)}</b>, one fit <b style="color:var(--bad)">${pct(s.pick.price)}</b> (of S).`;
    }

    function run() {
      const btn = $(".post-run");
      btn.disabled = true;
      btn.textContent = "Scoring…";
      const err = { pick: [], avg: [], prior: [] };
      let done = 0;
      const N = 1000;
      (function batch() {
        for (let i = 0; i < 50 && done < N; i++, done++) {
          const s = stock(false); // eval distribution: no forced spread
          // expected error of a random exact fit: averaged over all fits, less noisy than one draw
          err.pick.push(mean(s.fits.map((f) => ((f.price - s.truePrice) / STD) ** 2)));
          err.avg.push(((s.avg - s.truePrice) / STD) ** 2);
          err.prior.push(((s.priorMean - s.truePrice) / STD) ** 2);
        }
        btn.textContent = `Scoring… ${done}/${N}`;
        if (done < N) return setTimeout(batch, 0);
        const [p, a, pr] = [mean(err.pick), mean(err.avg), mean(err.prior)];
        $(".post-score").innerHTML = `
          <table>
            <tr><th>k = 1, standardised squared error</th><th class="num">this simulation</th><th class="num">our trained models (metrics.json)</th></tr>
            <tr><td>No context (prior mean)</td><td class="num">${pr.toFixed(4)}</td><td class="num">0.0571</td></tr>
            <tr><td style="color:var(--bad)">One exact fit (≈ calibration)</td><td class="num">${p.toFixed(4)}</td><td class="num">0.0211 (smile calibration)</td></tr>
            <tr><td style="color:var(--good)">Average of all fits (Bayes-optimal)</td><td class="num">${a.toFixed(4)}</td><td class="num">0.0107 (Transformer)</td></tr>
          </table>
          <p>Picking one fit costs ${(p / a).toFixed(1)}× the error of averaging (theory: exactly 2×). The errors are heavy-tailed, so a run of 1,000 can be off by about 10%; press again to see it move.</p>`;
        btn.disabled = false;
        btn.textContent = "Score 1,000 stocks";
      })();
    }

    $(".post-new").addEventListener("click", show);
    $(".post-run").addEventListener("click", run);
    show();
  }

  document.querySelectorAll(".posterior").forEach(build);
})();
