# Design Decisions and Comparisons

> Historical record: the numerical results below were produced on the original
> private course dataset. That provider-derived dataset is excluded from the
> public repository. These values document design exploration; they are not
> current benchmarks for the synthetic demo.

This document makes the project's main design choices explicit.

The goal is not just to show that the system runs, but to show that key choices
were recognized, compared, and justified.

## 1. Main decision points

| Decision point | Option A | Option B | Chosen default / use | Why this choice |
| --- | --- | --- | --- | --- |
| `stock_stock` sparsification | Correlation threshold | Top-k neighbors | Keep both; use threshold as a simple baseline, and use top-k for the interactive graph | Threshold is easy to explain. Top-k gives tighter control over graph density. |
| `stock_topic` edge weight | `article_count` | `avg_topic_relevance` | Default to `article_count` | Article count is more stable and easier to interpret as "topic exposure". |
| News layer in the main graph | Topic nodes | Article nodes | Topic nodes | Topic nodes keep the core graph compact and interpretable. |
| LLM integration | Offline enrichment | Per-query online inference | Offline enrichment for historical news; constrained online inference only for one new article | Offline enrichment is cheaper, faster at runtime, and easier to audit. |
| Graph stock universe | Any ticker mentioned in news | Only stocks with local price tables | Only stocks with price tables | This avoids partial nodes with missing price/sector data and keeps graph metrics meaningful. |

## 2. Comparison A: stock-stock edge construction

### Setup

- Universe: the dashboard's 15-ticker multi-sector core set
- News file: `merged_seed_news.json`
- Topic weight: `article_count`

The graph stays connected in every setting because the topic layer is dense.
So the real comparison is not "does the graph connect at all?" but:

- how much market-structure backbone is retained
- how readable the graph remains

### Results

| Strategy | Threshold | Top-k | `stock_stock` edges | `sector_sector` edges | Total edges | Avg degree | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Threshold only | 0.60 | None | 2 | 1 | 243 | 13.135 | Very sparse stock backbone |
| Threshold only | 0.70 | None | 1 | 0 | 241 | 13.027 | Too sparse for structure exploration |
| Top-k | 0.60 | 1 | 12 | 8 | 260 | 14.054 | Best balance for visual exploration |
| Top-k | 0.60 | 2 | 23 | 12 | 275 | 14.865 | Richer structure, but noticeably busier |

### Decision

- I keep **threshold mode** because it is a clean baseline and easy to explain.
- I use **top-k mode** for interactive exploration because it gives a much more usable market backbone.
- For the interactive graph page, I tightened the default to **top-k = 1** so the graph opens in a cleaner state.

### Why not just use threshold?

At this data scale, threshold-only mode quickly becomes too sparse at higher values.
The topic layer still keeps the graph connected, but the market-structure layer
becomes too weak to support interesting bridge analysis.

## 3. Comparison B: stock-topic weighting

### Setup

- Universe: the dashboard's 15-ticker multi-sector core set
- `stock_stock` setting: threshold `0.60`, top-k `1`

### Results

#### Default: `article_count`

Top topics by total `stock_topic` weight:

1. `financial_markets` = `17462`
2. `earnings` = `14187`
3. `technology` = `10884`
4. `finance` = `9709`
5. `economy_macro` = `4679`

Top `stock_topic` edges include:

- `JPM -> financial_markets` = `2439`
- `NVDA -> financial_markets` = `2310`
- `NVDA -> technology` = `2133`

Interpretation:

- this emphasizes repeated exposure
- strong topics are the ones that appear often
- the result is stable and easy to explain

#### Alternative: `avg_topic_relevance`

Top topics by total `stock_topic` weight:

1. `earnings` = `13.226`
2. `financial_markets` = `12.634`
3. `mergers_and_acquisitions` = `12.332`
4. `blockchain` = `12.277`
5. `life_sciences` = `12.077`

Top `stock_topic` edges include:

- `UNH -> blockchain` = `0.922`
- `UNH -> earnings` = `0.899`
- `JNJ -> life_sciences` = `0.898`

Interpretation:

- this emphasizes semantic tightness instead of volume
- niche topics can rank high from a smaller number of strongly matched articles
- this is useful for analysis, but less stable as a default exposure measure

### Decision

I keep **`article_count` as the default stock-topic weight** because it better
matches the project's main idea of topic exposure.

I still keep **`avg_topic_relevance` as a configurable alternative** because it
shows a different and legitimate interpretation of the same news layer.

## 4. Complexity and trade-off notes

These are the practical complexity decisions behind the design:

- Changing **threshold vs top-k** does **not** change the main cost of computing pairwise correlations. It mainly changes how many edges are retained after the correlation matrix already exists.
- Changing the **stock-topic weight column** is relatively cheap, because the candidate columns are already produced in `topic_stock`.
- Using **topic nodes instead of article nodes** changes scale materially. The project has `41,913` merged articles but only `15` topic labels in the current main graph layer. Article nodes would make the graph much larger and harder to interpret.
- Using **offline LLM enrichment** moves model cost to preprocessing time and keeps runtime behavior stable and reproducible.

## 5. Final design choices

These are the defaults selected after the original comparison:

- Keep **correlation threshold** as a simple baseline option.
- Use **top-k** when the goal is graph exploration, especially in the interactive graph.
- Default `stock_topic` weight to **`article_count`**.
- Keep **topic nodes**, not article nodes, in the main graph.
- Keep LLM as an **enrichment layer**, not a prediction layer.

## 6. Short summary

This project is not just "a graph that works."

It deliberately compares:

- two stock-backbone construction strategies
- two topic-weighting strategies
- two possible news-layer designs
- two LLM integration modes

The final system keeps the simpler, more stable defaults while still exposing
the alternative choices when comparison is useful.
