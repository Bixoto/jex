# ASSETS.md — visual index

Hand-authored SVG, no build dependencies. Every label and number in an asset traces to a
source in `FACT-BASE.md` / `publish/`; nothing is invented. Open the files directly or drop
them into the target repo.

**Where they go**

- **Engine repo (`github.com/J-I-B/jex`):** copy `assets/*.svg` to `assets/`. `ENGINE-README.md`
  references `assets/<name>.svg`.
- **HF dataset (`Bixoto/decision-index-results`):** copy `assets/*.svg` to `assets/` in the
  dataset repo; `DATASET-CARD.md` references `assets/<name>.svg`.

**Design system (shared)**

| token | value | use |
|---|---|---|
| surface | `#0B0E13` → `#07090C` | card background (dark "lab" theme) |
| panel | `#0E1218` / `#0C1015` | inner panels |
| ink | `#F4F6FA` / `#EDF0F5` | primary text |
| muted | `#8892A0` / `#B7C0CD` | secondary text |
| brand (Bixoto) | `#FF4200` | Jex, the adopted result, the hero accent |
| teal | `#34C9A2` | frozen / deterministic / verified |
| amber | `#E49B2D` | post-processing levers · latency gate |
| grey | `#4A5361` / `#5B6472` | trained peers, inert elements |
| display font | Sofia Pro (`--main` site family) → `Inter` → system-ui | headings |
| mono | SFMono / JetBrains Mono → Menlo/Consolas | every number |

Brand orange `#FF4200` and the Sofia Pro family are taken from
`bixoto_front/static/variables.scss` and `bixoto_front/static/fonts/`. The `b` mark in
`bixoto-mark.svg` is a geometric re-draw of `bixoto_front/static/favicon.png`
(the raster `bixoto_ai/static/logo-white.png` is a white wordmark, invisible on light and
unusable in a README, so the wordmark is rendered in SVG instead).

---

## Asset index

| path | type | purpose | data source (receipt) |
|---|---|---|---|
| `assets/hero.svg` | hero / banner | README header: title, one-line pitch, `60.29` public index, frozen `gemma-4-31B`, readout motif | `publish/VERIFY.md` §1 · `publish/DISCLOSURES.md` §1 |
| `assets/bixoto-mark.svg` | brand wordmark | footer / attribution, tasteful and minimal | `bixoto_front/static/favicon.png` · `variables.scss` (`--main: #ff4200`) |
| `assets/pipeline.svg` | architecture diagram | the engine end to end: state + typed questions → deterministic engine → single greedy step → 255-label readout → post-processing levers → answers + probabilities | `publish/PUBLISHED.md` §5 · `publish/PR-v1.xplus.md` |
| `assets/same-checkpoint.svg` | head-to-head diagram | the cleanest comparison on the board: one frozen checkpoint, deck31b 58.19 vs Jex 60.29, **+2.10** | `FACT-BASE.md` §B.1 · `COMPETITOR-CLAIMS.md` §2.5 (deck31b) |
| `assets/ladder.svg` | step/bar chart | the score ladder 59.48 → 59.96 → 60.09 → 60.21 → 60.29 + leave-one-outs | `publish/DISCLOSURES.md` §3 · `publish/jex-repo/LADDER.md` |
| `assets/public-index.svg` | comparison bar chart | public index on DI 0.3, trained vs frozen, with the frozen frontier line | `FACT-BASE.md` §A.1/§A.4 (board snapshot 2026-10-06) |
| `assets/latency.svg` | latency vs gate | median 116.4 / mean 369.6 / p80 319.9 ms vs the board's ≤ 1000 ms admission | `publish/jex-repo/LATENCY-CERTIFICATE.md` · `FACT-BASE.md` §A.4 |
| `assets/evidence-chain.svg` | process/receipts diagram | BI=1 serial → bit-exact re-score (two scorers) → receipts + rollback; stat strip | `publish/DISCLOSURES.md` §2.3 · `publish/VERIFY.md` §1–4 |
| `assets/dataset-cover.svg` | dataset cover | HF dataset card banner: run `jex-1.0`, `60.29`, complete, file tree | `publish/PUBLISHED.md` §1 |

---

## Notes

- All assets are 20 px-rounded dark cards, safe to place on a light or dark host page.
- Numbers shown use thin-space-free monospace for a "receipt" feel; every figure is quoted
  verbatim from the sources above.
- The **`+0.48 / +0.13 / +0.12 / +0.08`** deltas in `ladder.svg` are differences between
  consecutive published ladder values; they are not separate measurements.
- The `+2.10` in `same-checkpoint.svg` and `public-index.svg` is deck31b's public 58.19
  subtracted from Jex's public 60.29 (`FACT-BASE.md` §B.1).
- **Not shown anywhere, by design:** the internal BI-off latency mode, the rejected extras
  variant, private-store citations, and any Full-score or overall-rank claim.
