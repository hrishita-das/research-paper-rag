# RAG Error Analysis

## 1. Purpose

This document analyzes incorrect, partially correct, poorly grounded, or
insufficiently cited responses produced by the Research Paper RAG system.

The goal is to identify failure patterns in:

- Retrieval
- Reranking
- Answer generation
- Citation generation
- Out-of-domain detection
- Grounding

---

# 2. Evaluation Summary

Total evaluation questions: **25**

The evaluation dataset contains questions from multiple categories:

- Definition
- Process
- Equation
- Architecture
- Contribution
- Comparison
- Out-of-domain

The following metrics were obtained from `metrics.json`:

| Metric | Result |
|---|---:|
| Citation Validity | TODO |
| Answerable Success Rate | TODO |
| Out-of-Domain Refusal Rate | TODO |

---

# 3. Error Categories

## 3.1 Retrieval Errors

### Description

The required information exists in the document collection, but the relevant
chunk is not retrieved among the top-k candidates.

### Symptoms

- Relevant document exists but does not appear in retrieved results.
- Retrieved chunks discuss a related topic but do not answer the question.
- Low reranker score for the correct evidence.
- Correct answer cannot be generated because the evidence is missing.

### Possible Causes

- Poor semantic embedding.
- BM25 fails to match paraphrased queries.
- Chunk size is inappropriate.
- Important information is split across chunks.
- Query wording differs significantly from the paper wording.

### Example

**Question:**

> What is the main contribution of McCaD?

**Failure:**

The retriever returns general diffusion-model information but misses the
section describing McCaD's proposed methodology.

### Possible Fixes

- Improve chunking strategy.
- Increase candidate retrieval size.
- Improve hybrid retrieval weighting.
- Add query expansion.
- Experiment with a stronger embedding model.
- Improve reranker selection.

---

# 4. Reranking Errors

### Description

The relevant information is retrieved but ranked below irrelevant or
less-relevant chunks.

### Symptoms

- Correct document appears in the candidate set.
- An unrelated chunk receives a higher reranker score.
- The final top-k evidence does not contain the best supporting passage.

### Possible Causes

- Cross-encoder limitations.
- Short or ambiguous query.
- Similar terminology across different papers.
- Chunk contains insufficient context.

### Possible Fixes

- Test different cross-encoder models.
- Increase candidate_k.
- Increase rerank_k.
- Add metadata-aware reranking.
- Include neighboring chunks when appropriate.

---

# 5. Generation Errors

### Description

The correct evidence is retrieved, but the LLM produces an incorrect,
incomplete, or unsupported answer.

### Symptoms

- Retrieved evidence clearly contains the answer.
- Generated answer does not use the information correctly.
- Model adds information not present in the retrieved evidence.
- Model paraphrases an equation incorrectly.

### Possible Causes

- Weak prompt constraints.
- LLM instruction-following limitations.
- Excessive context.
- Ambiguous evidence.
- Mathematical formatting issues.

### Possible Fixes

- Strengthen the grounded system prompt.
- Use explicit evidence boundaries.
- Reduce unnecessary retrieved context.
- Add answer verification.
- Add claim-level grounding checks.

---

# 6. Citation Errors

### Description

The answer contains missing, invalid, or incorrectly assigned citations.

### Types

### 6.1 Missing Citation

The answer contains a factual statement but no `[Source N]` citation.

### 6.2 Invalid Citation

The answer references a source number that does not exist.

Example:

```text
The forward process adds Gaussian noise. [Source 7]

when only five sources were retrieved.

6.3 Incorrect Citation

The cited source exists but does not support the statement.

6.4 Citation Formatting Error

The information is correct but the citation format does not match the
required [Source N] format.

Possible Fixes
Keep citation instructions in the system prompt.
Validate every citation after generation.
Reject invalid citations.
Retry generation using a stricter citation prompt.
Implement claim-to-source verification.
7. Out-of-Domain Errors
Description

The user asks a question that cannot be answered using the indexed research
papers.

The system should refuse instead of using the LLM's pretrained knowledge.

Example

Question:

Who invented the telephone?

Expected behavior
The retrieved documents do not contain enough information to answer this question.
Important

The RAG system should not answer:

Alexander Graham Bell invented the telephone.

Even though this fact is known by the LLM, it is not supported by the
retrieved research-paper evidence.

Evaluation

The out-of-domain refusal rate measures whether the system correctly refuses
questions that cannot be answered from the document collection.

8. Equation Errors

Equations require special attention because small formatting or transcription
errors can change their meaning.

Common Problems
Missing square-root symbols.
Incorrect subscripts.
Missing variables.
Incorrect brackets.
Equation copied from memory instead of evidence.
Markdown/LaTeX corruption.
Example

Evidence:

q(xt|xt−1) = N(xt; √(1−βt)xt−1, βtI)

The generated answer should preserve the mathematical structure rather than
reconstructing the equation from general knowledge.

Recommended Fix

For equation questions:

Retrieve the relevant equation chunk.
Require the LLM to copy the equation from evidence.
Validate citation.
Avoid mathematical derivation unless explicitly requested.
9. Current Evaluation Errors

Record individual failures below.

Error 1

Question:
TODO

Category:
TODO

Expected Answer:
TODO

Generated Answer:
TODO

Retrieved Sources:
TODO

Error Type:
TODO

Root Cause:
TODO

Proposed Fix:
TODO

Error 2

Question:
TODO

Category:
TODO

Expected Answer:
TODO

Generated Answer:
TODO

Retrieved Sources:
TODO

Error Type:
TODO

Root Cause:
TODO

Proposed Fix:
TODO

Error 3

Question:
TODO

Category:
TODO

Expected Answer:
TODO

Generated Answer:
TODO

Retrieved Sources:
TODO

Error Type:
TODO

Root Cause:
TODO

Proposed Fix:
TODO

10. Error Analysis Table
ID	Category	Error Type	Retrieval	Generation	Citation	Severity	Proposed Fix
1	TODO	TODO	TODO	TODO	TODO	TODO	TODO
2	TODO	TODO	TODO	TODO	TODO	TODO	TODO
3	TODO	TODO	TODO	TODO	TODO	TODO	TODO
4	TODO	TODO	TODO	TODO	TODO	TODO	TODO
5	TODO	TODO	TODO	TODO	TODO	TODO	TODO
11. Error Distribution

After manually analyzing all failed samples, summarize the errors here.

Error Type	Count	Percentage
Retrieval	TODO	TODO
Reranking	TODO	TODO
Generation	TODO	TODO
Citation	TODO	TODO
Out-of-domain	TODO	TODO
Equation	TODO	TODO
12. Root Cause Analysis

The most important failure modes identified during evaluation are:

1. Retrieval

TODO

2. Reranking

TODO

3. Generation

TODO

4. Citation

TODO

5. Out-of-domain detection

TODO

13. Improvements Planned

Based on the evaluation, the following improvements will be considered:

 Improve document chunking.
 Experiment with chunk overlap.
 Tune dense/BM25 hybrid retrieval weights.
 Increase retrieval candidate size.
 Evaluate alternative embedding models.
 Evaluate alternative reranker models.
 Add query rewriting.
 Add query expansion.
 Improve citation validation.
 Add claim-level grounding verification.
 Improve equation handling.
 Add automated hallucination detection.
 Add retrieval metrics such as Recall@K.
 Add answer-quality evaluation using an LLM judge.
 Compare different LLMs.
14. Final Findings

The error analysis will be used to determine whether failures originate
primarily from retrieval, reranking, generation, or citation validation.

The objective is not only to improve the final answer accuracy, but to
identify which component of the RAG pipeline is responsible for each failure.

The final system should satisfy the following principles:

Relevant evidence should be retrieved.
Relevant evidence should be ranked highly.
Answers should be generated only from retrieved evidence.
Factual claims should have valid citations.
Unsupported questions should be rejected.
Equations should be preserved accurately.
The system should not rely on the LLM's pretrained knowledge when evidence
is unavailable.