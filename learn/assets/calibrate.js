// Calibration game: the reader plays the Transformer at k = 1.
// A stock has a hidden flat volatility; one example option's price is shown;
// the reader slides sigma until the model price matches, then the query option
// is priced at that sigma and scored against the truth and the no-context guess.
//
// Markup:  <div class="calib"></div>   (needs bs.js loaded first)

(function () {
  const SIG_LO = 0.1;
  const SIG_HI = 0.5; // the bs_flat prior, U[0.1, 0.5]
  const { callOverSpot } = window.BS;

  const pct = (p) => (100 * p).toFixed(2) + "% of S";
  const strike = (m) => (100 * Math.exp(m)).toFixed(0) + "% of S";
  const rand = (a, b) => a + (b - a) * Math.random();

  // price averaged over the prior: what a model that ignores the examples says
  function priorMean(m, T) {
    let s = 0;
    const n = 200;
    for (let i = 0; i < n; i++) s += callOverSpot(m, T, SIG_LO + ((i + 0.5) / n) * (SIG_HI - SIG_LO));
    return s / n;
  }
  const swing = (o) => callOverSpot(o.m, o.T, SIG_HI) - callOverSpot(o.m, o.T, SIG_LO);

  function informativeOption() {
    for (;;) {
      const o = { m: rand(-0.25, 0.25), T: rand(0.25, 2) };
      if (swing(o) > 0.04) return o;
    }
  }

  function build(el) {
    el.innerHTML = `
      <div class="calib-row"><span class="calib-label">Stock</span><span class="calib-stock"></span></div>
      <div class="calib-card">
        <p class="calib-h">Example option (seen in context)</p>
        <p>Strike <b class="ex-k"></b>, maturity <b class="ex-t"></b> years</p>
        <p>Observed price: <b class="ex-p"></b></p>
      </div>
      <label class="calib-slider">Your volatility guess σ = <b class="sig-v"></b>
        <input type="range" min="${SIG_LO}" max="${SIG_HI}" step="0.001">
      </label>
      <p class="calib-fit">Model price at your σ: <b class="fit-p"></b> <span class="fit-d"></span></p>
      <div class="calib-card">
        <p class="calib-h">Query option (price this)</p>
        <p>Strike <b class="q-k"></b>, maturity <b class="q-t"></b> years</p>
        <p>Your price: <b class="q-p"></b></p>
      </div>
      <p class="calib-buttons">
        <button type="button" class="lock">Lock in my price</button>
        <button type="button" class="new">New stock</button>
        <button type="button" class="hard">New stock, hard example</button>
      </p>
      <div class="calib-result"></div>`;

    const $ = (s) => el.querySelector(s);
    const slider = $("input");
    let state, count = 0;

    function newStock(hard) {
      count += 1;
      const ex = hard ? { m: rand(0.3, 0.42), T: rand(0.1, 0.18) } : informativeOption();
      let q;
      do q = informativeOption(); while (Math.abs(q.m - ex.m) < 0.08);
      state = { sigma: rand(SIG_LO, SIG_HI), ex, q, locked: false };
      state.exPrice = callOverSpot(ex.m, ex.T, state.sigma);
      $(".calib-stock").textContent = `#${count}, hidden σ drawn from U[0.10, 0.50]`;
      $(".ex-k").textContent = strike(ex.m);
      $(".ex-t").textContent = ex.T.toFixed(2);
      $(".ex-p").textContent = pct(state.exPrice);
      $(".q-k").textContent = strike(q.m);
      $(".q-t").textContent = q.T.toFixed(2);
      slider.value = 0.3;
      slider.disabled = false;
      $(".lock").disabled = false;
      $(".calib-result").innerHTML = "";
      update();
    }

    function update() {
      const s = +slider.value;
      const fit = callOverSpot(state.ex.m, state.ex.T, s);
      const d = fit - state.exPrice;
      $(".sig-v").textContent = s.toFixed(4);
      $(".fit-p").textContent = pct(fit);
      const close = Math.abs(d) < 0.0005;
      $(".fit-d").textContent = close ? "matches" : d > 0 ? "too high: lower σ" : "too low: raise σ";
      $(".fit-d").className = "fit-d " + (close ? "ok" : "off");
      $(".q-p").textContent = pct(callOverSpot(state.q.m, state.q.T, s));
    }

    function lock() {
      const s = +slider.value;
      const { q, sigma } = state;
      const truth = callOverSpot(q.m, q.T, sigma);
      const mine = callOverSpot(q.m, q.T, s);
      const prior = priorMean(q.m, q.T);
      const errMine = Math.abs(mine - truth);
      const errPrior = Math.abs(prior - truth);
      const ratio = errPrior / Math.max(errMine, 1e-7);
      const exSwing = swing(state.ex);
      slider.disabled = true;
      $(".lock").disabled = true;
      $(".calib-result").innerHTML = `
        <p><b>Hidden σ was ${sigma.toFixed(4)}</b>; you said ${s.toFixed(4)}.</p>
        <table>
          <tr><th>Query price</th><th class="num">value</th><th class="num">error</th></tr>
          <tr><td>True</td><td class="num">${pct(truth)}</td><td class="num">—</td></tr>
          <tr><td>Yours (calibrated from 1 example)</td><td class="num">${pct(mine)}</td><td class="num">${(100 * errMine).toFixed(3)}</td></tr>
          <tr><td>No-context guess (prior mean)</td><td class="num">${pct(prior)}</td><td class="num">${(100 * errPrior).toFixed(3)}</td></tr>
        </table>
        <p>${
          ratio > 3
            ? `You beat the no-context guess about ${ratio.toFixed(0)}×. That gap is what in-context learning looks like.`
            : `You barely beat the no-context guess.`
        } ${
          exSwing < 0.01
            ? `The example option moved only ${(100 * exSwing).toFixed(2)}% of S across the whole σ range, so its price says almost nothing about σ.`
            : ""
        }</p>`;
    }

    slider.addEventListener("input", update);
    $(".lock").addEventListener("click", lock);
    $(".new").addEventListener("click", () => newStock(false));
    $(".hard").addEventListener("click", () => newStock(true));
    newStock(false);
  }

  document.querySelectorAll(".calib").forEach(build);
})();
