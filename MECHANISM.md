# Two-bidder reference construction

Research context: Dütting et al., *Mechanism Design for Large Language Models*,
arXiv:2310.10826v3, especially Sections 3.3 and 4.2 [1]. The concrete common-mass
coupling below is derived here for the restricted two-bidder case.

## Inputs and output distribution

At one fixed generated prefix, take two normalized distributions `p_A`, `p_B`
over the same token IDs and nonnegative bids `a`, `b`, with `a+b > 0`.
The required output distribution is:

```text
lambda = a / (a+b)
q = lambda*p_A + (1-lambda)*p_B
```

Bids must not alter either distribution at this fixed prefix.

## Sampling and payments

Define tokenwise common mass `c[t] = min(p_A[t], p_B[t])`, its total `C=sum(c)`,
and disagreement mass `d=1-C=TV(p_A,p_B)`.

With probability `C`, sample from `c/C`. The outcome is independent of both bids;
charge zero to both bidders. Skip this branch when `C=0`.

Otherwise, sample candidates independently from

```text
r_A = (p_A-c)/d
r_B = (p_B-c)/d
```

Skip this branch when `d=0`. The residual supports are disjoint. Draw an
independent `u` uniformly in `(0,1)`:

```text
u < lambda:  emit A's candidate; A pays b*u/(1-u); B pays zero
otherwise:   emit B's candidate; B pays a*(1-u)/u; A pays zero
```

The same random tape is reused when inspecting a counterfactual bid at the same
prefix, with the same distributions. Do not confuse this with a full regenerated
answer, where later prefixes and distributions generally change.

## Why this works

The common branch contributes `c[t]` to the unconditional output probability.
The contested branch contributes

```text
d * (lambda*r_A[t] + (1-lambda)*r_B[t])
= lambda*(p_A[t]-c[t]) + (1-lambda)*(p_B[t]-c[t])
```

Their sum is exactly `q[t]`.

For a fixed tape, both residual candidates and `u` are fixed. Increasing A's bid
can only switch the output from B's candidate to A's candidate, at the threshold
`a = b*u/(1-u)`. Increasing B's bid gives the reverse switch at
`b = a*(1-u)/u`. A's candidate is in the support where `p_A > p_B`; B's is where
`p_B > p_A`. Thus the construction is stable for both bidders in this restricted
case. A winner pays its critical bid, never more than its submitted bid.

This is not a highest-bid-wins auction: at bids `(1,1)` and `u=.4`, A wins and pays
`2/3`, not the rival's bid of `1`.

## Expected-payment test oracle

For A, with rival bid `b>0`, integrate the threshold over winning random draws:

```text
E[charge_A] = d * integral_0^lambda [b*u/(1-u)] du
            = d*b*[log(1+a/b) - a/(a+b)]
```

The analogous formula exchanges A and B. If one bidder has zero bid, neither pays.
For identical distributions both always pay zero. For orthogonal distributions
and bids `(1,1)`, each expected charge is `log(2)-.5`, which is positive.

**Paper-check note:** the total-variation expression printed in Proposition 3.13
appears to have the distance terms reversed: increasing the bid reduces distance,
so the printed order can give a negative charge. The threshold definition and
token-probability integral instead give the positive formula above. Treat this
as an apparent typo, not an author-confirmed erratum. The tests use the threshold
definition as the oracle [1, pp. 16–17].

## Scope and nonclaims

This is a construction for two fixed distributions at one token step. It is not
an implementation of arbitrary-bidder stable sampling. It does not infer a
monetary valuation from a probability, protect against strategic prompt/model
selection, or prove incentives over complete generated text.

Zero/zero bids are rejected by the auction core: route them to a separate no-ad
baseline. Use simulated credits. Floating-point arithmetic is not a production
payment ledger. Account caps, refunds, filtering, reserves, and adaptive bids
require additional mechanism analysis.

## Experimental winner continuation (October 14, 2026)

The live interface also offers a user-requested extension, enabled by default
with three extra tokens and a 98% confidence threshold. Set extra tokens to zero
to use the reference construction above. The auction core and its tests are
unchanged; the extension changes how its decisions are used across tokens.

After a **contested** A or B win at step `s`, grant that winner an allowance of
up to `L` immediately following tokens (`L=3` by default). At each new prefix:

1. Recompute both full-vocabulary distributions using the configured temperature.
2. If the previous winner assigns a token probability **at least** `h` (`h=.98`
   by default), emit that token deterministically, attribute it to that winner,
   charge both bidders zero, and consume one allowance token.
3. At the first confidence miss, or after `L` carried tokens, expire the allowance
   and run a fresh reference auction at that same step. A fresh contested win
   grants a new allowance; a shared outcome grants none. Carried tokens never
   renew an allowance. The threshold must be greater than .5, so at most one
   token qualifies.

Append every chosen ID, including a carried EOS, to both advertiser contexts and
the read-only no-ad observer. EOS, the safety limit, and manual stop still apply.
No separate ad-free answer is generated. The explicit zero-bid baseline never
uses continuation.
Settings and bids remain fixed during the round and never enter model prompts.

On a carried step the actual output distribution is a point mass, **not** the
bid-weighted mixture `q`. Events and the inspector retain `p_A`, `p_B`, and the
hypothetical auction `q`, and separately report actual output probability. Each
carried event records the original win's step, ordinal, confidence, threshold,
remaining allowance, and zero charges. No auction tape is drawn for that step;
the next fresh auction consumes the next tape. The inspector links back to the
original win instead of presenting a current-step bid counterfactual that would
ignore the history-dependent entitlement.

This is a heuristic to keep high-confidence continuations together, not a claim
that a phrase or message will finish correctly. Confidence is a next-token
probability after temperature, not a factual-accuracy score. It changes the
sequence distribution and grants additional influence for free. The reference
mixture identity, local incentive analysis, and expected-payment formulas do not
establish guarantees for this extended sequence mechanism. Model hallucinations
and unsuccessful completions remain visible and unedited.

## Reference

[1] Paper: https://arxiv.org/abs/2310.10826
