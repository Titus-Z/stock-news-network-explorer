# Architecture and Design Decisions

## Project

**Stock and News Network Explorer**

This project builds a graph-based exploration system for financial markets.
The graph is the primary structure of the system:

- `stock` nodes
- `sector` nodes
- `topic` nodes

Core edges:

- `stock_stock`
- `stock_sector`
- `stock_topic`
- `sector_sector`

The project is not a stock prediction system and not an investment recommendation tool.

## Main design decisions

### 1. `stock_stock` sparsification

Two alternatives were kept:

- correlation threshold
- top-k neighbors

Why this matters:

- without sparsification, the stock layer becomes too dense
- threshold is easier to explain
- top-k gives tighter control over graph readability

Final choice:

- keep both options
- use threshold as a simple baseline
- use tighter `top-k` defaults in the interactive graph

### 2. `stock_topic` edge weighting

Two main alternatives were compared:

- `article_count`
- `avg_topic_relevance`

Why this matters:

- `article_count` measures repeated exposure
- `avg_topic_relevance` measures semantic tightness

Final choice:

- default to `article_count`
- keep `avg_topic_relevance` as a configurable alternative

### 3. News layer design

Two alternatives:

- topic nodes
- article nodes

Final choice:

- use topic nodes in the main graph

Reason:

- topic nodes are much more compact and interpretable
- article nodes would greatly increase graph size and reduce readability

### 4. LLM integration strategy

Two alternatives:

- offline enrichment
- real-time LLM calls for every query

Final choice:

- offline enrichment for historical news
- constrained one-shot analysis only for one new article

Reason:

- lower runtime cost
- lower latency
- better reproducibility
- easier manual checking

## Comparison results

### Comparison A: `threshold` vs `top-k`

Using the dashboard's default 15-stock universe:

| Strategy | `stock_stock` edges | `sector_sector` edges | Total edges | Avg degree |
| --- | ---: | ---: | ---: | ---: |
| threshold `0.60` | 2 | 1 | 243 | 13.135 |
| threshold `0.70` | 1 | 0 | 241 | 13.027 |
| top-k `1` | 12 | 8 | 260 | 14.054 |
| top-k `2` | 23 | 12 | 275 | 14.865 |

Interpretation:

- threshold-only mode becomes too sparse for market-structure exploration
- `top-k = 1` gives the best balance between structure and readability
- `top-k = 2` is richer but noticeably more crowded

### Comparison B: `article_count` vs `avg_topic_relevance`

Key finding:

- `article_count` produces a more stable and intuitive topic-exposure view
- `avg_topic_relevance` highlights smaller but tighter semantic matches

Interpretation:

- `article_count` is better as the default
- `avg_topic_relevance` is useful as an alternative analytical lens

## Complexity and trade-offs

- switching between threshold and top-k does not change the main cost of computing correlations; it mainly changes how many edges are retained
- switching the `stock_topic` weight column is cheap because the candidate columns already exist in the aggregated `topic_stock` table
- using topic nodes instead of article nodes is a major scale decision
- offline LLM enrichment moves model cost to preprocessing and makes runtime behavior more stable

## Final defaults defended by the project

- keep threshold mode as a baseline
- use top-k for cleaner graph exploration
- default `stock_topic` weight to `article_count`
- keep topic nodes, not article nodes, in the core graph
- keep LLM as an enrichment layer, not a prediction layer

## Why this project is not just a default pipeline

This project does not only implement one fixed system. It explicitly:

- exposes multiple graph-construction options
- compares alternative settings
- justifies final defaults
- explains trade-offs between readability, stability, and scale

That is the main design contribution of the project.
