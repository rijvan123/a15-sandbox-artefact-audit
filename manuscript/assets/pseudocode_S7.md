**Algorithm 1. Seal an evidence item**

1. Input source file f, installation c and provenance p; compute d_f = SHA-256(f) and compare it with the published digest; on mismatch, quarantine and stop.
2. Parse report r; split it into trace record x, environment record e and label record y; strip label-bearing fields from x.
3. Compute d_x = SHA-256(canon(x)) and d_e = SHA-256(canon(e)).
4. Form entry(i) = (i, ID, d_f, d_x, d_e, "acquire", HMAC_k(actor), τ_i, h(i−1)) and set h(i) = SHA-256(entry(i) ‖ h(i−1)).
5. Append entry(i) to the ledger and write x, e and y to the write-once store.
6. At batch end, compute R_b = MerkleRoot(h(j), …, h(n)), obtain an RFC 3161 token on R_b and append an anchor entry.

**Algorithm 2. Freeze the lexicon and analysis plan**

1. From training-period reports only, derive P rules from ATT&CK T1497 sub-techniques and CAPEv2 documentation, and F rules from value classes the environment sets.
2. List boundary tokens with both readings.
3. Write the lexicon, H1–H4, decision margins, family thresholds, alias table and split manifests.
4. Hash each file into the ledger, anchor the batch with an RFC 3161 token and publish the registration.
5. Accept a test score only if τ(anchor) < τ(first test-scoring entry).

**Algorithm 3. Build harmonised token sequences**

1. Map each call to t = (API, argument classes, return class); drop raw values.
2. Replace APIs outside the common hook set H_c with one out-of-set token.
3. Apply the documented curation filter to both installations; keep an early-exit flag.
4. Interleave processes by call time, collapse repeats with a run-length bucket and cut to length L (head or head-plus-tail).
5. Tag each token B, F or P with the frozen lexicon.
6. Remove test traces whose MinHash Jaccard similarity to any training trace exceeds the training-fixed threshold; repeat after masking.

**Algorithm 4. Masking audit and mitigation**

1. For each learner, seed and origin, train the base model on unmasked tokens and the mitigated model with F masked plus artefact dropout at rate p.
2. Score both models on within- and cross-installation test sets under test masks none, F, P, F ∪ P and matched random.
3. Train the length-only, artefact-only and environment-ID probes on the same splits.
4. Compute macro-F1 and per-family recall; estimate 95% cluster-bootstrap percentile intervals from 9,999 replicates, with seeds averaged.
5. Apply the registered H1–H4 decision rules, combining installations by the intersection–union rule, with Holm adjustment; record every number in the ledger.

**Algorithm 5. Score a casework trace**

1. Seal the trace with Algorithm 1 and build its sequence with Algorithm 3, F tokens masked.
2. Compare its environment record with the validated set on CAPE version, guest build, hook-set digest, timeout and network route.
3. Predict the family distribution with the mitigated model and keep the top three.
4. Compute Integrated Gradients class shares; refer the trace if the F share exceeds θ_F.
5. Attach the within-installation error rate if the environment matches, otherwise the cross-installation rate and an analyst referral.
6. Write the finding record, append its ledger entry and add the Merkle path and token to the report.

