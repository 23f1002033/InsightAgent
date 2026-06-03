"""
eval
────
RAGAS evaluation harness for InsightAgent (Phase 7).

Metrics computed:
  faithfulness       — does the answer stick to the retrieved context?
  answer_relevancy   — is the answer on-topic for the question?
  context_precision  — are retrieved chunks relevant?
  context_recall     — does the retrieved context cover the ground truth?
"""
