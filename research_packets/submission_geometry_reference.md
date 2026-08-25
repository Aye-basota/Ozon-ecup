# Submission geometry — external reference

Submission geometry is a separate TEST-only research line. It is not part of
model training, feature generation, canonical validation, or the offline baseline.

## External workspace

- Root: `C:/Users/Admin/Desktop/submission_geometry_research`
- Current incumbent artifact:
  `submission_geometry/SUBMIT_NEXT_BEST.csv`
- Current incumbent public score: `1.6466079084`
- Artifact SHA256:
  `95f3fa982d8173e5382b199888748f57601b86fe8d0dcaa692984dde67d34677`
- Previous confirmed geometry artifact:
  `current_best/SUBMIT_v2_shrunk.csv`, public `1.6467120249048954`, SHA256
  `50f8d11dca25c782d42f66ebd46ba50a71a533188ca5f20bad4178cb005b74aa`.

The `1.6466079084` score is supplied by the user and bound to the intended next
candidate above. There is no fold-safe OOF equivalent for this 65-source geometry
result; never synthesize one from TEST predictions or public scores.

Historical pre-geometry teammate components remain in:
`C:/Users/Admin/Desktop/OZON-E-CUP/пайплайн сокомандника/latest/`.
They are references only; they are not copied here.

## PASS handoff from model research

After a model experiment receives `PASS` under its predeclared OOF gate:

1. Generate `artifacts/test/EXP_XXX_NAME.parquet` in this repository.
2. Verify `user_id` matches `sample_submit.csv`, predictions are finite and
   non-negative after `expm1`, and record the artifact SHA256.
3. Export a two-column `user_id,predict` CSV in sample order and place a copy in
   the external geometry workspace's `submissions/` directory together with the
   experiment ID, OOF metrics, family, and SHA256.
4. Let the geometry pipeline ingest it as a new source. Do not copy the geometry
   submission bank, fitted geometry cache, or geometry reports back into this repo.
5. Record any submitted result in `registry/submissions.csv`; keep the canonical
   offline baseline unchanged unless a fold-safe OOF baseline actually improves.
