## Notation (draft of Supplementary Table S3)

S7's symbols are kept: I, s, t = (a, κ, r), x, L, V, H_c, Λ, c(t), B, F, P, M_S(x), M_rand(x), p, y, K, f_θ(x), F1_within, F1_cross, G, ρ, Δ, IG_j(x), x′, m, φ_C(x), θ_F, J, τ_J, e, E_val, h(i), H, ‖, R_b, TST_b, τ_i, HMAC_k, α, N_boot and n_s. The one change of form is J(𝒜, 𝒜′) for S7's J(A, B). Symbols new in S8:

| Symbol | Meaning | Equation |
| --- | --- | --- |
| tok(·); ⟨OOS⟩, ⟨UNK⟩, ⟨M⟩ | Token map; out-of-set, unknown and mask tokens | 1, 3 |
| κ(args), r(ret) | Argument-class tuple; return class | 1 |
| n_call; t₍₁₎ to t₍n_call₎ | Number of calls; tokens in call-time order | 2 |
| n_run, β(·), β_∅ | Run length; run-length bucket; neutral bucket of a masked element | 2, 3 |
| trunc_L | Head-plus-tail truncation to L elements | 2 |
| ℳ_S(x) | Positions of x whose class is in S | 3 |
| ξ_j; p* | Dropout indicator; selected dropout rate | 3 |
| cell(j), cnt_cell, ℛ | Frequency band and position bin of position j; cell counts; random position set | 4 |
| emb(·), emb_β(·), pos(·) | Token, bucket and position embeddings | 5 |
| U, d, d_h, n_head | Layer input; model width 128; head width 32; 4 heads | 5 |
| W_q, W_k, W_v, W_out | Projection matrices | 5, 6 |
| Π | Padding mask | 5 |
| ū, z(x), ŷ | Mean-pooled output; logits; predicted family | 6 |
| ℒ(θ), w_k, γ | Training loss; class weight; class-weight exponent | 7 |
| n_k^tr, n̄^tr | Training traces of family k; mean family count | 7 |
| 𝒜(x), ω | API-name 3-gram set of x; one 3-gram (an n-gram in S1) | 8, S1 |
| μ_l, n_h, Ĵ | Min-wise hash functions; signature length; MinHash estimate | 8 |
| 𝒟^tr, 𝒟^val, 𝒟^te, 𝒟^te,dd | Training, validation, test and de-duplicated test sets | 9 |
| ∼ | Near-duplicate relation among a family's test traces | 9 |
| Q_C^(I), J_w, TV | Class-C token frequencies of installation I; weighted Jaccard; total variation distance | 10 |
| TP_k, FP_k, FN_k | Confusion counts of family k | 11 |
| Prec_k, Rec_k, F1_k, F1_macro | Precision, recall and F1 of family k; macro-F1 in points | 11 |
| π_k, Γ_kk′ | Family share of a test set; rate of predicting k′ for a trace of family k | 12 |
| F1_k^bal, v_n | Family-balanced F1; balancing weight of trace n | 12 |
| FAR_k, LR_k | False-attribution rate; likelihood ratio of a family-k finding | 12 |
| O, o, 𝒦_o | Number of origins (3 for MELD-DS-448); origin; families scored at origin o | 13 |
| F̄1 | Seed-averaged family-balanced macro-F1 | 13 |
| f^base, f^mit, f^art, f^len | Base, mitigated, artefact-only and length-only models | 13–15 |
| ψ₁, ψ₂, ψ₃, ψ_4a, ψ_4b | Effects for H1–H4 | 15 |
| g, 𝒰_g, u | Bootstrap stratum (test set, origin, family); its clusters; one cluster | 9, 16 |
| ν, 𝒟^*ν, ψ̂^*ν | Replicate index; resampled test sets; replicate estimate | 16 |
| σ | Seed index, 1 to n_s | 16 |
| p^>(ψ; ψ₀), p^<(ψ; ψ₀), ψ₀ | Bootstrap P values for ψ above and below ψ₀; boundary value | 17 |
| p_j^sup, p_j^rej, p₂^eq | Support, rejection and equivalence P values | 18, 20 |
| p̃, m_H | Holm-adjusted P value; Holm family size, 4 | 19 |
| emb_λ, λ, λ_l, q_l | Path point; path parameter; quadrature nodes and weights | 21 |
| δ(x) | Completeness gap | 22 |
| φ_F^base | Footprint share from the base model on the unmasked trace | 24 |
| ref(x), match(e), e^(I), η_F | Referral indicator; environment match; validated record; referral budget 0.05 | 24 |
| quant_{1−η_F} | Empirical (1 − η_F) quantile | 24 |
| enc(·), entry(i), n_led | Canonical encoding; ledger entry; number of entries | 25 |
| MTH, 𝐡, n_L | Merkle tree hash; list of chain hashes; size of the left subtree | 26 |
| i_b^s, i_b^e, n_b, T_b, Sign_TSA | First and last entry of batch b; batch size; authority time; authority signature | 26 |
| ℰ, ℰ′, Verify, Adv_H^coll, n_q | Ledger; altered ledger; verification; collision advantage; hash evaluations | 27 |
| i_frz, i_anc, i_1, V_frz | Freeze entry; its batch anchor; first scoring entry; ordering indicator | 28 |
| tfidf_ω, tf_ω, df_ω, idf_ω, N^tr | TF-IDF weight, count, document frequency and inverse document frequency; training traces | S1 |
| Obj, CE, z^cur, f^new | Boosting objective; cross-entropy; current logits; new trees | S2 |
| n_leaf,k, χ_ℓ, ζ, λ₂ | Leaves; leaf value; minimum split gain; L2 penalty | S2 |
| grad_nk, hess_nk, Grad_ℓ, Hess_ℓ, Gain | Per-trace derivatives; leaf sums; split gain | S3 |
| n_band, n_row, P_cand, J₅₀ | LSH bands and rows; candidate probability; 50% point | S4 |
| N_ρ, G_pool | Pooled numerator and pooled gap of ρ | S5 |
| hw₉₅, z₀.₉₇₅, c_F, deff, N_te, N_req | Interval half-width; normal quantile; macro-F1 variance factor; design effect; test traces; required traces | S6 |
| f̄, n₁₀, n₀₁, P_McN | Seed-ensemble posterior; discordant counts; McNemar P value | S7 |
| HL, diff_k, δ_Cliff, U_MW | Hodges–Lehmann estimate; per-family difference; Cliff's δ; Mann–Whitney statistic | S8 |
| Y₁, Y₂, n₁, n₂ | Values and sizes of two compared groups | S8 |
| AUT, N_mon, F1_l | Area Under Time; monthly slots; macro-F1 in slot l | S9 |

l, l′, k and k′ are summation indices only.

