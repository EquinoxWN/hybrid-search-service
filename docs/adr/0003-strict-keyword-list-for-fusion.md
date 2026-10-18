# ADR 0003: Require every query term in the keyword list that feeds fusion

- **Status:** Accepted

## Context

`websearch_to_tsquery` joins the query's terms with AND. On the 35 judged natural-language
queries, keyword search then found a relevant document in the top 10 for only 4 queries
(nDCG@10 0.073): a document rarely contains every word of "pause the program for a while".
Joining the terms with OR instead (any term matches, ranked by `ts_rank_cd`) is the usual way to
raise keyword recall, so it was implemented and measured on the same index and queries.

| Keyword query | Keyword nDCG@10 | Keyword hits | Hybrid nDCG@10 | Hybrid MRR@10 | Hybrid hits |
|---|---|---|---|---|---|
| AND (`websearch_to_tsquery`) | 0.073 | 4/35 | **0.471** | **0.432** | **31/35** |
| OR of the same terms | 0.208 | 16/35 | 0.405 | 0.392 | 28/35 |

Vector search alone: nDCG@10 0.463, MRR@10 0.404, 30/35.

## Decision

Keep AND. The keyword list is used as a precise signal for fusion, not as a recall channel.

## Consequences

- With AND, the keyword list is often empty, and hybrid then equals vector search; when all terms
  match (identifiers, exact phrases such as "create a temporary directory"), it adds a strong,
  usually correct candidate.
- With OR, common words ("string", "data", "file") pull in dozens of loosely related documents
  whose fused scores push relevant vector results down: "convert a string to an integer" returned
  `flag.String` and `strings.Builder.String` first.
- Keyword mode on its own stays weak for natural-language questions; that is a measured property
  of the mode, reported as such.
- The sample is 35 queries written by the author. M2's 200 judged queries, and a cross-encoder
  over the top 50, may change this; the comparison above is the baseline to repeat.
