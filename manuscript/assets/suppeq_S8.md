## 8 Supplementary Equations S1–S9

**TF-IDF weights (n-gram LightGBM).** For an n-gram ω of 1–3 tokens,

$$
\operatorname{tfidf}_\omega(x)=\frac{\mathrm{tf}_\omega(x)\,\mathrm{idf}_\omega}{\big\lVert\mathrm{tf}(x)\odot\mathrm{idf}\big\rVert_2},\qquad \mathrm{idf}_\omega=\ln\frac{1+N^{\mathrm{tr}}}{1+\mathrm{df}_\omega}+1\qquad(\mathrm{S}1)
$$

tf_ω(x) is the raw count, df_ω the number of training traces containing ω and N^tr the number of training traces; df is fitted on training traces only. scikit-learn's TfidfVectorizer with smoothed idf and L2 norm reproduces this to 1.1 × 10⁻¹⁶ (check C07).

**Boosting objective.** Each LightGBM round adds one tree per family to the current logits z^cur:

$$
\mathrm{Obj}=\sum_{n\in\mathcal D^{\mathrm{tr}}}\mathrm{CE}\big(y_n,\ z^{\mathrm{cur}}(x_n)+f^{\mathrm{new}}(x_n)\big)+\sum_{k=1}^{K}\Big[\zeta\,n_{\mathrm{leaf},k}+\frac{\lambda_2}{2}\sum_{\ell\in\mathrm{leaves}_k}\chi_\ell^2\Big]\qquad(\mathrm{S}2)
$$

CE is the softmax cross-entropy of equation (6), f^new the new trees, n_leaf,k the number of leaves in family k's tree, χ_ℓ the value of leaf ℓ, ζ the minimum split gain and λ₂ the L2 penalty on leaf values.

**Gradients, leaf values and split gain.** With p_nk the current softmax probability,

$$
\begin{aligned}\mathrm{grad}_{nk}&=p_{nk}-\mathbf 1[y_n=k],\qquad \mathrm{hess}_{nk}=\frac{K}{K-1}\,p_{nk}(1-p_{nk}),\qquad \chi^{*}_\ell=-\frac{\mathrm{Grad}_\ell}{\mathrm{Hess}_\ell+\lambda_2}\\ \mathrm{Gain}&=\frac12\left[\frac{\mathrm{Grad}^2_{\mathrm{left}}}{\mathrm{Hess}_{\mathrm{left}}+\lambda_2}+\frac{\mathrm{Grad}^2_{\mathrm{right}}}{\mathrm{Hess}_{\mathrm{right}}+\lambda_2}-\frac{(\mathrm{Grad}_{\mathrm{left}}+\mathrm{Grad}_{\mathrm{right}})^2}{\mathrm{Hess}_{\mathrm{left}}+\mathrm{Hess}_{\mathrm{right}}+\lambda_2}\right]-\zeta\end{aligned}\qquad(\mathrm{S}3)
$$

Grad_ℓ and Hess_ℓ sum grad and hess over the traces in leaf ℓ, and "left" and "right" are the children of a candidate split. Minimising Grad·χ + ½(Hess + λ₂)χ² in each leaf gives the leaf value and the gain (check C06). The factor K/(K − 1) is LightGBM's own: its source (multiclass_objective.hpp, master branch, read on 1 November 2026) sets factor_ to num_class/(num_class − 1) and multiplies p(1 − p) by it.

**LSH candidate probability.**

$$
P_{\mathrm{cand}}(J)=1-\big(1-J^{\,n_{\mathrm{row}}}\big)^{n_{\mathrm{band}}},\qquad J_{50}=\big(1-2^{-1/n_{\mathrm{band}}}\big)^{1/n_{\mathrm{row}}},\qquad n_h=n_{\mathrm{band}}\,n_{\mathrm{row}}\qquad(\mathrm{S}4)
$$

One band of n_row hashes agrees with probability J^{n_row}, and a pair becomes a candidate if any of the n_band bands agrees (check C13).

| Bands × rows | J₅₀ | P at J = 0.3 | P at 0.5 | P at 0.7 | P at 0.8 | P at 0.9 |
| --- | --- | --- | --- | --- | --- | --- |
| 32 × 4 | 0.383 | 0.229 | 0.873 | 0.9998 | > 0.99999 | > 0.99999 |
| 25 × 5 | 0.487 | 0.059 | 0.548 | 0.990 | 0.99995 | > 0.99999 |
| 21 × 6 | 0.565 | 0.015 | 0.282 | 0.928 | 0.998 | > 0.99999 |
| 16 × 8 | 0.674 | 0.001 | 0.061 | 0.613 | 0.947 | 0.99988 |

**Delta method for ρ.** With N_ρ the pooled numerator of equation (14) and G_pool the pooled gap,

$$
\operatorname{Var}(\hat\rho)\approx\frac{\operatorname{Var}(N_\rho)-2\rho\operatorname{Cov}(N_\rho,G_{\mathrm{pool}})+\rho^2\operatorname{Var}(G_{\mathrm{pool}})}{G_{\mathrm{pool}}^2}\qquad(\mathrm{S}5)
$$

This only checks the bootstrap interval of ρ. It breaks down as G_pool approaches zero, which is why H4 is tested only once H3's gap is supported. Check C15: simulated standard deviation 0.0584 against 0.0578 at a 15-point gap and ρ = 0.40.

**Sample size for the K5 check.** For a gap between two independent test sets of N_te shared-family traces each, with mean recall and design effect deff,

$$
\mathrm{hw}_{95}\approx z_{0.975}\sqrt{\frac{2\,c_F^2\,\mathrm{deff}\,\overline{\mathrm{Rec}}\,(1-\overline{\mathrm{Rec}})}{N_{\mathrm{te}}}},\qquad N_{\mathrm{req}}=\frac{2\,c_F^2\,\mathrm{deff}\,z_{0.975}^2\,\overline{\mathrm{Rec}}\,(1-\overline{\mathrm{Rec}})}{\Delta^2}\qquad(\mathrm{S}6)
$$

The macro-recall variance, Σ_k Rec_k(1 − Rec_k)/n_k divided by K², stands in for the macro-F1 variance, and c_F is the ratio of their standard deviations: 1.00 and 1.01 in the two simulated confusion structures (check C16). A first-order argument explains the near-identity. Since FN_k = n_k − TP_k, F1_k = 2TP_k/(TP_k + n_k + FP_k). When a family receives about as many predictions as it has traces, the derivative of F1_k with respect to TP_k and minus its derivative with respect to FP_k add up to 1/n_k; and because the FP_k sum to N − Σ TP_k, macro-F1 moves with total recall. N_req is 3,074 at mean recall 0.8 and deff = 1, 4,610 at deff = 1.5, 4,034 at recall 0.7 and 1,729 at recall 0.9. At N_te = 3,000 the half-width is 2.04 points. Averaging over independent origin windows divides the within-installation variance by O; an undated installation's single test set does not.

**Seed ensemble and exact McNemar test.**

$$
\bar f(x)=\frac{1}{n_s}\sum_{\sigma=1}^{n_s}f_{\theta_\sigma}(x),\qquad P_{\mathrm{McN}}=\min\Big\{1,\ 2\sum_{l=0}^{\min(n_{10},\,n_{01})}\binom{n_{10}+n_{01}}{l}\,2^{-(n_{10}+n_{01})}\Big\}\qquad(\mathrm{S}7)
$$

n₁₀ counts traces that model A attributes correctly and model B does not, and n₀₁ the reverse, using the seed-ensemble predictions argmax f̄(x). The test compares top-1 accuracy, not macro-F1, so it stays secondary (S6). Check C20 matches SciPy's exact binomial test.

**Effect sizes.**

$$
\mathrm{HL}=\operatorname{median}\Big\{\frac{\mathrm{diff}_k+\mathrm{diff}_{k'}}{2}:\ k\le k'\Big\},\qquad \delta_{\mathrm{Cliff}}=\Pr(Y_1>Y_2)-\Pr(Y_1<Y_2)=\frac{2\,U_{\mathrm{MW}}}{n_1n_2}-1\qquad(\mathrm{S}8)
$$

diff_k is the paired per-family difference behind the Wilcoxon test across families (S6). Y₁ and Y₂ are values from two groups of sizes n₁ and n₂, and U_MW is the Mann–Whitney statistic of the first group, with ties counted as one half (check C21).

**Area Under Time.**

$$
\mathrm{AUT}=\frac{1}{N_{\mathrm{mon}}-1}\sum_{l=1}^{N_{\mathrm{mon}}-1}\frac{\mathrm{F1}_l+\mathrm{F1}_{l+1}}{2}\qquad(\mathrm{S}9)
$$

This is TESSERACT's normalised trapezoid area over N_mon monthly slots. Scoring the origin-1 model (trained to February 2025) on March–August 2025 gives N_mon = 6. Nothing is tuned on those months, so the curve is a decay summary for Supplementary Information (check C22).

