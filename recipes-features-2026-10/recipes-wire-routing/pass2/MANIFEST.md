# #15030 pass2: wires on today's main

- Upstream PR: unslothai/unsloth#15030, head `LeoBorcherding:recipes-wire-routing` @ `34f9dab1f32bef7cad77f156719211d50d79451e` (untouched).
- AFTER: local merge of that head into main `856799b427` = `cbfb52a0807ec98aee786ac962213d1a28188100`, pushed only as the throwaway fork branch `shots/recipes-wires-on-main`.
- Merge conflicts, resolved locally (all under `studio/frontend/src/features/recipe-studio/`): `stores/recipe-studio.ts`, `utils/graph/derive-display-graph.ts`, `utils/handles.ts`, `utils/layout.ts`, `utils/payload/types.ts`. git rerere replayed the earlier resolutions of the same hunks (mostly comments main had shortened next to the PR's changes; the PR side kept for code). Gate on the merge: `node scripts/typecheck.mjs` clean, `node --test tests/recipe*.test.ts` 58/58.
- BEFORE: unslothai/unsloth main `856799b4272ed679c727620c43c320954e9419a1` (2026-10-29), pushed to the fork as `shots/recipes-base-main`.
- Machine: Colab T4 (Tesla T4, 15 GB), Linux, VM `fl-t4-nst-recipes-shots-86097` (night mode, tag `nst-recipes-shots`, stopped after the run; logs in `COLAB LOGS EXPORT/pr-15030/fl-t4-nst-recipes-shots-86097/20261030-030554-final`).
- How: one real Unsloth Studio backend booted from main (`setup.sh --local`). Both PRs are frontend-only, so the two AFTER frontends were built on the VM (`npm ci && npm run build`) and served by the same backend by swapping `index.html`. Playwright headless Chromium 1440x900 @2x, logged in on the VM; theme set with PUT `/api/settings/personalization` plus `prefers-color-scheme`. Recipes were PUT through `/api/data-recipe/recipes` and opened in full view.
- Images are 900 px wide, light (no suffix) and dark (`-dark`). `*-before-after*.png` stack BEFORE over AFTER with labelled bars naming both SHAs. GIFs are light only, 12 fps.
- pr-media commit with every image: `f5d7bfde228d68480e42eebff1288ff3394584cf` (all URLs below verified 200 at that SHA).

## Scenes

| scene | files | notes |
|---|---|---|
| Text to SQL, before/after | 01, 02, 01-02 composite | each build's own `learning-recipes/text-to-sql.json` (the PR moves the two model cards) |
| 21-block support recipe after Auto layout | 03, 04, 03-04 composite | framed on tier, sla, ticket, reply; main's Auto layout places the cards differently, so the frames differ |
| whole support recipe after Auto layout, with legend | 05 (after), 05b (before), 05 composite | Center view; both stop at React Flow's 0.5 minimum zoom |
| hops and junction dots | 06 | around `sla` |
| hover highlight | 07 gif | hovering issue, sla, reply in turn |
| drag a card, reroute on drop | 08 gif | `sla` dragged down and back |
| Reroute dropped on a wire, then dragged | 11 gif | 11-block support-tickets recipe, plan_tier -> ticket wire |
| recipe saved top to bottom (new) | 12, 13, 12-13 composite | the support-tickets recipe saved with `layout_direction: TB` and top/bottom handles |

Not reshot: `09/10-stress-scattered` (on pr-media, not in the body; its recipe is not in `polished/`).

- [01-02-learning-sql-before-after-dark.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-wire-routing/pass2/01-02-learning-sql-before-after-dark.png)
- [01-02-learning-sql-before-after.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-wire-routing/pass2/01-02-learning-sql-before-after.png)
- [01-learning-sql-before-main-dark.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-wire-routing/pass2/01-learning-sql-before-main-dark.png)
- [01-learning-sql-before-main.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-wire-routing/pass2/01-learning-sql-before-main.png)
- [02-learning-sql-after-dark.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-wire-routing/pass2/02-learning-sql-after-dark.png)
- [02-learning-sql-after.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-wire-routing/pass2/02-learning-sql-after.png)
- [03-04-support-before-after-dark.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-wire-routing/pass2/03-04-support-before-after-dark.png)
- [03-04-support-before-after.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-wire-routing/pass2/03-04-support-before-after.png)
- [03-support-before-main-dark.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-wire-routing/pass2/03-support-before-main-dark.png)
- [03-support-before-main.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-wire-routing/pass2/03-support-before-main.png)
- [04-support-after-dark.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-wire-routing/pass2/04-support-after-dark.png)
- [04-support-after.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-wire-routing/pass2/04-support-after.png)
- [05-support-overview-after-dark.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-wire-routing/pass2/05-support-overview-after-dark.png)
- [05-support-overview-after.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-wire-routing/pass2/05-support-overview-after.png)
- [05-support-overview-before-after-dark.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-wire-routing/pass2/05-support-overview-before-after-dark.png)
- [05-support-overview-before-after.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-wire-routing/pass2/05-support-overview-before-after.png)
- [05b-support-overview-before-main-dark.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-wire-routing/pass2/05b-support-overview-before-main-dark.png)
- [05b-support-overview-before-main.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-wire-routing/pass2/05b-support-overview-before-main.png)
- [06-hop-junction-closeup-dark.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-wire-routing/pass2/06-hop-junction-closeup-dark.png)
- [06-hop-junction-closeup.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-wire-routing/pass2/06-hop-junction-closeup.png)
- [07-hover-highlight-bus.gif](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-wire-routing/pass2/07-hover-highlight-bus.gif)
- [08-drag-card-reroute-on-drop.gif](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-wire-routing/pass2/08-drag-card-reroute-on-drop.gif)
- [11-reroute-drop-and-drag.gif](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-wire-routing/pass2/11-reroute-drop-and-drag.gif)
- [12-13-tb-recipe-before-after-dark.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-wire-routing/pass2/12-13-tb-recipe-before-after-dark.png)
- [12-13-tb-recipe-before-after.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-wire-routing/pass2/12-13-tb-recipe-before-after.png)
- [12-tb-recipe-before-main-dark.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-wire-routing/pass2/12-tb-recipe-before-main-dark.png)
- [12-tb-recipe-before-main.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-wire-routing/pass2/12-tb-recipe-before-main.png)
- [13-tb-recipe-after-dark.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-wire-routing/pass2/13-tb-recipe-after-dark.png)
- [13-tb-recipe-after.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-wire-routing/pass2/13-tb-recipe-after.png)

## Findings

1. A recipe saved top to bottom opens with left/right ports but keeps its saved top-to-bottom positions, so wires leave each card sideways and climb over the row above (13). It only becomes a left-to-right layout after Auto layout.
2. The port legend pill is semi-transparent and sits over the canvas; wires and card text show through it (06, 13).
3. Seen on main too: `customer_name`'s formula `{{ customer.first_name }} ...` is outlined red as invalid, yet a preview run fills that column (12, and #15037's 07).
