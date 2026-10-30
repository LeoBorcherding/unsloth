# #15037 pass2: super cards on today's main

- Upstream PR: unslothai/unsloth#15037, head `LeoBorcherding:recipes-super-cards` @ `0bad4bad7fc5f17d3bed318dca47e84261f86659` (untouched).
- AFTER: local merge of that head into main `856799b427` = `a1edccbe6711cb3bbda183a1caa1c340bbed7829`, pushed only as the throwaway fork branch `shots/recipes-super-on-main`.
- Merge conflicts, resolved locally (all under `studio/frontend/src/features/recipe-studio/`): `recipe-studio-page.tsx`, `stores/recipe-studio.ts`, `utils/payload/types.ts`. git rerere replayed the earlier resolutions (comment-only hunks next to the PR's additions; the PR's code kept). Gate on the merge: typecheck clean, `node --test tests/recipe*.test.ts` 45/45.
- BEFORE: unslothai/unsloth main `856799b4272ed679c727620c43c320954e9419a1` (2026-10-29), pushed to the fork as `shots/recipes-base-main`.
- Machine: Colab T4 (Tesla T4, 15 GB), Linux, VM `fl-t4-nst-recipes-shots-86097` (night mode, tag `nst-recipes-shots`, stopped after the run; logs in `COLAB LOGS EXPORT/pr-15030/fl-t4-nst-recipes-shots-86097/20261030-030554-final`).
- How: one real Unsloth Studio backend booted from main (`setup.sh --local`). Both PRs are frontend-only, so the two AFTER frontends were built on the VM (`npm ci && npm run build`) and served by the same backend by swapping `index.html`. Playwright headless Chromium 1440x900 @2x, logged in on the VM; theme set with PUT `/api/settings/personalization` plus `prefers-color-scheme`. Recipes were PUT through `/api/data-recipe/recipes` and opened in full view.
- Images are 900 px wide, light (no suffix) and dark (`-dark`). `*-before-after*.png` stack BEFORE over AFTER with labelled bars naming both SHAs. GIFs are light only, 12 fps.
- pr-media commit with every image: `f5d7bfde228d68480e42eebff1288ff3394584cf` (all URLs below verified 200 at that SHA).

## Scenes

The 11-block support-tickets recipe from pass1 (`polished/demo-recipe.json`), grouped into "Customer persona" (customer, customer_name, plan_tier, mood) and "Ticket fields" (product_area, severity, sla_target, subject).

| scene | files | notes |
|---|---|---|
| select four blocks (new before/after) | 01, 01-before-main, 01 composite | main shows the same selection with no Combine action |
| select and combine | 02 gif | the new card is still "Super card 1" here; it is renamed afterwards in the breadcrumb |
| collapsed card | 03 | |
| inside, breadcrumb | 04 | |
| open and back | 05 gif | double-click in, breadcrumb out |
| both cards in context (new before/after) | 06, 06-before-main, 06 composite | after Auto layout and Center view |
| last run | 07 | a real 5-row Test run on the VM's backend: "Preview completed, 4/4 columns" |
| after Ungroup | 08 | the four blocks back, no overlaps |

- [01-selection-before-after-dark.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-super-cards/pass2/01-selection-before-after-dark.png)
- [01-selection-before-after.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-super-cards/pass2/01-selection-before-after.png)
- [01-selection-before-main-dark.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-super-cards/pass2/01-selection-before-main-dark.png)
- [01-selection-before-main.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-super-cards/pass2/01-selection-before-main.png)
- [01-selection-dark.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-super-cards/pass2/01-selection-dark.png)
- [01-selection.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-super-cards/pass2/01-selection.png)
- [02-select-and-combine.gif](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-super-cards/pass2/02-select-and-combine.gif)
- [03-collapsed-card-dark.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-super-cards/pass2/03-collapsed-card-dark.png)
- [03-collapsed-card.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-super-cards/pass2/03-collapsed-card.png)
- [04-inside-breadcrumb-dark.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-super-cards/pass2/04-inside-breadcrumb-dark.png)
- [04-inside-breadcrumb.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-super-cards/pass2/04-inside-breadcrumb.png)
- [05-open-and-back.gif](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-super-cards/pass2/05-open-and-back.gif)
- [06-recipe-before-main-dark.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-super-cards/pass2/06-recipe-before-main-dark.png)
- [06-recipe-before-main.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-super-cards/pass2/06-recipe-before-main.png)
- [06-two-cards-before-after-dark.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-super-cards/pass2/06-two-cards-before-after-dark.png)
- [06-two-cards-before-after.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-super-cards/pass2/06-two-cards-before-after.png)
- [06-two-cards-context-dark.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-super-cards/pass2/06-two-cards-context-dark.png)
- [06-two-cards-context.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-super-cards/pass2/06-two-cards-context.png)
- [07-last-run-dark.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-super-cards/pass2/07-last-run-dark.png)
- [07-last-run.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-super-cards/pass2/07-last-run.png)
- [08-after-ungroup-dark.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-super-cards/pass2/08-after-ungroup-dark.png)
- [08-after-ungroup.png](https://raw.githubusercontent.com/LeoBorcherding/unsloth/f5d7bfde228d68480e42eebff1288ff3394584cf/recipes-features-2026-10/recipes-super-cards/pass2/08-after-ungroup.png)

## Findings

1. The floating "Combine 4 blocks into a super card" button sits over the unselected `mood` card's header (01).
2. Wires around a card still use main's handles (03, 06), as the body says, since #15030 is not merged.
