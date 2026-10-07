# In-context option pricing: Resources

All entries were checked on 2026-10-06 (URL fetched or metadata confirmed via Crossref/RePEc; see notes).

## Knowledge

### The paper we reproduce
- [Paper: Garg, Tsipras, Liang, Valiant (2022), "What Can Transformers Learn In-Context? A Case Study of Simple Function Classes", arXiv:2208.01066](https://arxiv.org/abs/2208.01066)
  The foundation of the whole project. Use for: the prompt setup (§2), linear regression vs least squares (§3), distribution shift (§4), other function classes (§5), curriculum and capacity (§6). Abstract: "performance comparable to the optimal least squares estimator."
- [Code: dtsip/in-context-learning](https://github.com/dtsip/in-context-learning)
  The authors' code and released models; our fork's upstream. Use for: checking what the public code does vs the paper (they differ on some baselines).

### Black–Scholes and option pricing
- [Paper: Black & Scholes (1973), "The Pricing of Options and Corporate Liabilities", JPE 81(3):637–654](https://doi.org/10.1086/260062)
  The original formula. Use for: citing the formula in the paper; the fact that "the expected return on the stock does not appear" in it.
- [Paper: Merton (1973), "Theory of Rational Option Pricing", Bell J. Econ. 4(1):141–183](https://doi.org/10.2307/3003143)
  The companion derivation; cite alongside Black–Scholes. Metadata only verified.
- [Lecture: MIT OCW 18.S096 (2013), Lecture 19 "Black-Scholes Formula, Risk-neutral Valuation" (Vasily Strela)](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/resources/lecture-19-black-scholes-formula-risk-neutral-valuation/)
  Free video plus notes. Use for: the best beginner route into the formula and risk-neutral pricing.
- [Book: Hull, *Options, Futures, and Other Derivatives*, 11th ed. (Pearson, 2021)](https://www.pearson.com/en-us/subject-catalog/p/options-futures-and-other-derivatives/P200000005938/9780136939917)
  The standard textbook; ch. 20 "Volatility smiles and Volatility Surfaces". Use for: textbook citations on implied volatility, the smile and moneyness conventions.
- [Lecture notes: Derman, "The Volatility Smile", Columbia E4718, Lecture 1](https://emanuelderman.com/wp-content/uploads/2013/09/smile-lecture1.pdf)
  By a leading practitioner (hosted on his own site). Use for: why real smiles exist ("Since the crash, the volatility surface of index options has become skewed") and why flat vol is the Black–Scholes assumption.
- [Paper: Breeden & Litzenberger (1978), "Prices of State-Contingent Claims Implicit in Option Prices", J. Business 51(4):621–651](https://doi.org/10.1086/296025)
  Call prices across strikes determine the risk-neutral distribution. Use for: extension B only. Metadata only verified.

### Why in-context learning works (theory)
- [Paper: Xie, Raghunathan, Liang, Ma (2021), "An Explanation of In-context Learning as Implicit Bayesian Inference", arXiv:2111.02080](https://arxiv.org/abs/2111.02080)
  ICL as inferring a latent concept shared by the examples. Use for: framing our task as inferring a latent σ.
- [Paper: Raventós, Paul, Chen, Ganguli (2023), "Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression", arXiv:2306.15063](https://arxiv.org/abs/2306.15063)
  With low task diversity the Transformer behaves "like a Bayesian estimator with the … pretraining task distribution as the prior." Use for: explaining why our model beats least squares at small k.
- [Paper: Panwar, Ahuja, Goyal (2023), "In-Context Learning through the Bayesian Prism", arXiv:2306.04891](https://arxiv.org/abs/2306.04891)
  "high-capacity transformers mimic the Bayesian predictor"; also covers mixtures of function families. Use for: the posterior-mean baseline argument and extension A (mixed pricing models).
- [Paper: Akyürek et al. (2022), "What learning algorithm is in-context learning? Investigations with linear models", arXiv:2211.15661](https://arxiv.org/abs/2211.15661)
  Trained in-context learners match GD, ridge and least squares. Use for: related work.
- [Paper: von Oswald et al. (2022), "Transformers learn in-context by gradient descent", arXiv:2212.07677](https://arxiv.org/abs/2212.07677)
  The mechanistic view (GD in the forward pass). Use for: related work only (mechanisms are out of scope).

### Monte Carlo methods
- [Book chapter: Owen, *Monte Carlo theory, methods and examples*, ch. 9 "Importance sampling"](https://artowen.su.domains/mc/Ch-var-is.pdf) (book page: https://artowen.su.domains/mc/)
  Free, by a Stanford statistician. §9.2 self-normalised importance sampling; §9.3 effective sample size, eq. (9.13). Use for: the posterior-mean baseline and its reliability diagnostic.

## Wisdom (Communities)
- [Quantitative Finance Stack Exchange](https://quant.stackexchange.com/)
  Use for: sanity-checking the finance side (smile parameterisation, moneyness ranges, whether a reviewer from finance would accept the setup).
- [Cross Validated](https://stats.stackexchange.com/)
  Use for: the posterior-mean baseline and the evaluation statistics (bootstrap CIs, seeds).
- [EleutherAI Discord](https://discord.gg/eleutherai) (linked from eleuther.ai)
  Active research community on LLMs and interpretability. Use for: feedback on the ICL framing.
- [r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
  Broad; lower signal. Use for: visibility once the paper is out, not for detailed feedback.
- Local: supervisor / KTH research group. Likely the highest-value feedback; to confirm with Harriet.

## Gaps
- No verified short, free source for the smile *parameterisation* we use (quadratic in log-moneyness). We need one to cite or justify as a stylised choice.
- No quotable source checked for Breeden–Litzenberger's result (only metadata). Fetch the paper before citing the claim.
