# IMPLEMENTATION_CONTRACT.md - Binding Agreement: Code Must Match DECISIONS.md

## ⚠️ CRITICAL: GIT INSTRUCTIONS
**DO NOT RUN ANY GIT COMMANDS** (git add, git commit, git push, or any git modification commands).
See DECISIONS.md for full git warning. Work directly on files; user handles git operations.

---

## The Contract

**This document is a binding agreement between the agent building this system and the user requesting it.**

### Core Rule
**The actual implementation MUST NOT differ from DECISIONS.md.**

Every architectural decision documented in DECISIONS.md section 4.X is **law**. The code must conform to those decisions.

---

## What This Means in Practice

### ✓ Permitted: Implement Exactly as Documented

```python
# ✓ OK - Matches DECISIONS.md section 4.2 (quantile regression)
class QuantileRegressionLSTM(nn.Module):
    def __init__(self, quantiles=[0.05, 0.50, 0.95, 0.99]):
        super().__init__()
        self.quantiles = quantiles  # As documented
        self.n_quantile_heads = len(quantiles)
        # ... rest of init
```

### ✗ Not Permitted: Change Decision Without Updating DECISIONS.md First

```python
# ✗ NOT OK - Skipped quantile regression, used only p50
# This violates DECISIONS.md section 4.2
class QuantileRegressionLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.predict_median_only = True  # ← UNDOCUMENTED CHANGE!
        # ... rest of init
```

### ✗ Not Permitted: Simplify Without Documenting Trade-off

```python
# ✗ NOT OK - Removed region weighting because "too complex"
# Without updating DECISIONS.md first!
loss = mean_quantile_pinball_loss(y_true, y_pred)
# No region weighting → violates DECISIONS.md section 4.4
```

### ✓ Permitted: Simplify WITH Documentation

**If implementation is too complex, follow this process:**

1. **Identify the problem**: "Region weighting is complex, causing X issue"
2. **Document in DECISIONS.md**: Update section 4.4
   ```
   ### Decision 4.4: Loss Function - Region-Weighted Multi-Quantile
   
   | Aspect | Details |
   |--------|---------|
   | **Status** | ⚠ BLOCKED - Region weighting too complex |
   | **Problem** | [Describe why it's failing] |
   | **Proposed alternative** | [Simpler approach] |
   | **Evidence** | [Why simpler approach is acceptable] |
   ```
3. **Get approval**: User reviews and approves (or rejects) simplification
4. **Implement**: Now implement the simplified version
5. **Test & validate**: Measure if performance degrades
6. **Final entry**: Update DECISIONS.md section 6 with results

---

## Deviations Require Documentation

### When Code Must Deviate from DECISIONS.md

| Scenario | Required Action |
|----------|-----------------|
| Architecture too complex | Document problem in DECISIONS.md, propose simpler alternative, get approval |
| Implementation hits bug | Document bug in DECISIONS.md, document workaround, explain why necessary |
| Performance significantly worse | Document in DECISIONS.md, propose change, measure impact |
| Design assumption fails | Document failure in DECISIONS.md, explain why, try alternative |
| Data quality issue discovered | Document in DATA_INVESTIGATION.md, update DECISIONS.md section 7 |
| Library version incompatible | Document in REPRODUCIBILITY.md, update pyproject.toml |

### Example: Documented Deviation

**Scenario**: Region weighting formula is too complex to implement efficiently.

**Step 1: Document the problem in DECISIONS.md**
```markdown
### Decision 4.4: Loss Function - Region-Weighted Multi-Quantile

| Aspect | Details |
|--------|---------|
| **Options** | (A) Region-weighted pinball loss (complex, compute-heavy) |
| | (B) Simpler stratified sampling (no explicit weighting) |
| **Chosen** | (B) Simpler approach |
| **Original choice** | (A) was planned |
| **Deviation reason** | Computing regional loss aggregation in PyTorch DistributedDataParallel |
| | caused 3× slowdown and memory issues. Stratified sampling |
| | (ensuring each batch has ~25% per region) achieves similar fairness. |
| **Evidence** | Tested: Stratified sampling produces per-region val loss variance of 1.1×, |
| | close to 1.0× target. Region imbalance still mitigated. |
| **Trade-off** | Slightly less precise weighting, but 3× faster training. |
| **Status** | ✓ APPROVED (implementation simpler, fairness preserved) |
```

**Step 2: Implement the simpler version**
```python
# Stratified batch sampling (simpler than region weighting in loss)
def get_stratified_batch(data, batch_size=32):
    """Ensure ~batch_size/4 samples per region."""
    regions = ['Bhatagaon', 'IGKV', 'AIIMS', 'SILTARA']
    batch = []
    for region in regions:
        region_data = data[data['region'] == region]
        samples = np.random.choice(len(region_data), batch_size // 4, replace=True)
        batch.extend(region_data.iloc[samples])
    return np.array(batch)
```

**Step 3: Validate fairness still works**
```python
# Check per-region val loss variance
per_region_loss = {r: compute_loss_for_region(r) for r in regions}
loss_values = list(per_region_loss.values())
variance_ratio = max(loss_values) / min(loss_values)
print(f"Per-region loss variance: {variance_ratio:.2f}× (target: <1.5×)")
assert variance_ratio < 1.5, "Fairness degraded beyond acceptable threshold"
```

**Step 4: Document results in DECISIONS.md section 6**
```
### 4.4 Implementation Results

Stratified sampling (simpler approach):
- Per-region val loss variance: 1.08× ✓ (within target <1.5×)
- Training speed: 3× faster than region-weighted loss
- Per-region RMSE ratio: 1.2× (acceptable fairness)

Conclusion: Simplification successful. Fairness preserved, efficiency improved.
```

---

## The Source of Truth

### DECISIONS.md is Law

At any point in the project, if there's conflict between:
- **Code implementation** vs **DECISIONS.md**
- **Proposed change** vs **DECISIONS.md**
- **Bug/issue** vs **DECISIONS.md**

**DECISIONS.md wins.** Code must be updated to match DECISIONS.md, or DECISIONS.md must be updated first (with documentation).

### Example: Code Review

**Code review finds:**
```python
# In train_lstm.py
seq_len = 24  # Fixed sequence length
# ✗ VIOLATES DECISIONS.md section 4.5: "Adaptive: seq_len = 2 × horizon"
```

**Resolution:**
1. Either: Fix code to use adaptive sequence length (implement DECISIONS.md)
2. Or: Update DECISIONS.md to document why fixed seq_len is necessary, get approval, then implement

**Not permitted:** Commit code that contradicts DECISIONS.md without updating DECISIONS.md first.

---

## Updating DECISIONS.md: The Process

### When to Update

Update DECISIONS.md BEFORE implementation if:
- Architecture is unclear
- Design decision needs refinement
- Implementation strategy not documented
- Trade-offs not explicit

Update DECISIONS.md DURING implementation if:
- Blocked by issue (document obstacle)
- Implementation is harder than expected
- Performance significantly different

Update DECISIONS.md AFTER implementation if:
- Training completes (document results)
- Evaluation finishes (document metrics)
- Phase completes (document lessons learned)

### How to Update

**Format for each decision:**
```markdown
### Decision 4.X: [Title]

| Aspect | Details |
|--------|---------|
| **Options** | (A) Option A description |
| | (B) Option B description |
| **Chosen** | (A) Option A |
| **Rationale** | - Reason 1 |
| | - Reason 2 |
| **Implementation** | [Code/config details] |
| **Evidence** | None yet / [Results from testing] |
| **Status** | ✓ APPROVED / ◯ IN PROGRESS / ✗ REJECTED |
```

**Status codes:**
- ✓ **APPROVED**: Decision finalized, ready for implementation
- ◯ **TODO**: Pending investigation/decision
- ◯ **IN PROGRESS**: Currently being implemented/tested
- ✗ **REJECTED**: Alternative chosen instead
- ⚠ **BLOCKED**: Needs external input or resolution

---

## Transparency Requirements

### What Must Be Made Explicit

1. **Every hyperparameter value**
   - Not "learning rate is reasonable"
   - But: "learning_rate = 0.001 (see ARCHITECTURE.md section 5.1)"

2. **Every preprocessing choice**
   - Not "handle missing values"
   - But: "Interpolate gaps <6h, break on >6h (see PREPROCESSING_STRATEGY.md section 3.2)"

3. **Every architectural change from original design**
   - Document why original design didn't work
   - Show evidence for alternative
   - Update DECISIONS.md with deviation

4. **Every performance result**
   - Not "model trains fine"
   - But: "LSTM CRPS = 12.3 µg/m³ (target <15), converged at epoch 47"

### Transparency Violations (Not Permitted)

- ✗ Changing code without updating documentation
- ✗ Using different hyperparameters than documented
- ✗ Implementing without explaining why (if deviating from DECISIONS.md)
- ✗ Hiding negative results or failures
- ✗ Assuming defaults that aren't documented

### Transparency Examples

**✓ Good:**
```python
# From DECISIONS.md section 4.4: Region-weighted loss
# Calculate weight per region: weight_r = (1/4) / fraction_r
region_weight = {
    'Bhatagaon': 2.84,   # (1/4) / 0.088 = 2.84
    'IGKV': 0.44,        # (1/4) / 0.57 = 0.44
    'AIIMS': 1.26,       # (1/4) / 0.198 = 1.26
    'SILTARA': 1.27,     # (1/4) / 0.197 = 1.27
}
```

**✗ Bad:**
```python
# No reference, no explanation
region_weight = {
    'Bhatagaon': 3.0,
    'IGKV': 0.5,
    'AIIMS': 1.2,
    'SILTARA': 1.3,
}
```

---

## Code Review Checklist

Before any code is considered complete, verify:

- [ ] Code matches DECISIONS.md specifications
- [ ] All deviations documented in DECISIONS.md (with rationale)
- [ ] Hyperparameters reference section of DECISIONS.md or ARCHITECTURE.md
- [ ] Preprocessing matches PREPROCESSING_STRATEGY.md
- [ ] No future data leakage (features all have lag ≥ 0)
- [ ] Test set held out (never used during training)
- [ ] Random seeds set at script start
- [ ] Library versions match pyproject.toml
- [ ] Preprocessing outputs saved with metadata
- [ ] Training logs capture all hyperparameters
- [ ] Results documented in DECISIONS.md section 6
- [ ] Visualizations match VISUALIZATIONS.md specs
- [ ] Per-region fairness validated (if applicable)
- [ ] No git commands run

---

## Binding Commitment

**As the implementing agent, I commit to:**
- Never implement code that contradicts DECISIONS.md
- Always update DECISIONS.md before deviating from documented design
- Document every deviation with rationale and evidence
- Be transparent about failures, trade-offs, and changes
- Validate that implementation matches documented design
- Maintain DECISIONS.md as single source of truth

**If I encounter a situation where DECISIONS.md is unclear or incomplete:**
- I will update DECISIONS.md FIRST (with explicit documentation of the gap)
- I will propose alternative approaches if implementation is infeasible
- I will wait for approval before proceeding
- I will NOT silently work around documented constraints

---

## Escalation Process

**If implementation conflicts with DECISIONS.md:**

1. **Stop implementation**
2. **Document the conflict in DECISIONS.md**
   - Section 4.X: Mark decision as ⚠ BLOCKED
   - Add "Problem" field explaining conflict
   - Propose alternative or simplification
3. **Request feedback** (implicitly via user review)
4. **Wait for approval before continuing**
5. **Implement approved approach**
6. **Document results and update status to ✓ APPROVED or ✗ REJECTED**

---

## Final Audit

At project completion, the following must be true:

1. **Code and DECISIONS.md match**: Every architectural choice in code can be traced to DECISIONS.md
2. **No undocumented decisions**: No code implements design choices not in DECISIONS.md
3. **All deviations documented**: Any code differences from original design explained in DECISIONS.md
4. **Results recorded**: All performance metrics in DECISIONS.md section 6
5. **Reproducibility verified**: Another agent can run pipeline and get identical results
6. **Visualizations complete**: All outputs match VISUALIZATIONS.md specs

