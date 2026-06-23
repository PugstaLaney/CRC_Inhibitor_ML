# ChEMBL Cheat Sheet

Everything you actually need to write ChEMBL bioactivity queries. The full schema PDF (~80 tables) is overkill — these 5 tables cover 95% of real work.

## The 5 tables that matter

| Table | Primary key | What it stores | Joins to |
|---|---|---|---|
| `target_dictionary` | `tid` | Proteins / cell lines. One row per biological target. | `assays.tid` |
| `assays` | `assay_id` | Experiment metadata (assay type, confidence, target). | `activities.assay_id`, `target_dictionary.tid` |
| `activities` | `activity_id` | The actual measurements (one row per IC50 / Ki / etc). | `assays.assay_id`, `molecule_dictionary.molregno` |
| `molecule_dictionary` | `molregno` | Molecules with their ChEMBL IDs. | `activities.molregno`, `compound_structures.molregno` |
| `compound_structures` | `molregno` | SMILES + InChI for each molecule. | `molecule_dictionary.molregno` |

## How they connect

```
                  activities (act)
                  ┌─────────────────┐
                  │ assay_id        │──────► assays (a) ───► tid ──► target_dictionary (td)
                  │ molregno        │
                  │ standard_value  │
                  │ standard_units  │
                  │ pchembl_value   │
                  │ standard_type   │
                  └────────┬────────┘
                           │
                           └─► molecule_dictionary (md) ──► compound_structures (cs)
                                  (linked by molregno)
```

Read each arrow as "follow this link to get more info." A JOIN in SQL is just an arrow you're walking down.

## Canonical bioactivity query (template)

This is the query pattern for "give me every measurement against these targets, with SMILES." Save this — every bioactivity query you ever write is a mutation of it.

```sql
SELECT
    md.chembl_id            AS molecule_chembl_id,   -- which compound
    cs.canonical_smiles,                              -- ML input
    act.standard_type,                                -- IC50 / Ki / EC50 / ...
    act.standard_relation,                            -- = / > / <
    act.standard_value,                               -- the number
    act.standard_units,                               -- nM / uM / ...
    act.pchembl_value,                                -- pre-computed -log10(M)
    a.assay_type,                                     -- B / F / A / T / P
    a.confidence_score,                               -- 0-9
    td.chembl_id            AS target_chembl_id,     -- which target
    td.pref_name            AS target_name
FROM activities act
JOIN assays              a  ON act.assay_id    = a.assay_id
JOIN target_dictionary   td ON a.tid           = td.tid
JOIN molecule_dictionary md ON act.molregno    = md.molregno
JOIN compound_structures cs ON md.molregno     = cs.molregno
WHERE td.chembl_id IN ('CHEMBL2189121')               -- swap target(s) here
  AND act.standard_type   = 'IC50'                    -- swap measurement type here
  AND act.standard_value  IS NOT NULL
  AND cs.canonical_smiles IS NOT NULL
```

## Common variations

### Single target, IC50

Change the `IN (...)` to one ChEMBL ID:

```sql
WHERE td.chembl_id = 'CHEMBL203'   -- EGFR
```

### Multiple targets

Comma-separated list:

```sql
WHERE td.chembl_id IN ('CHEMBL2189121', 'CHEMBL5145', 'CHEMBL203', 'CHEMBL4005')
```

### Both IC50 and Ki (more data, less consistent)

```sql
AND act.standard_type IN ('IC50', 'Ki')
```

### Human only (drop animal/bacterial assays)

```sql
AND td.organism = 'Homo sapiens'
```

### High-quality measurements only (tighter filter)

```sql
AND act.standard_relation = '='
AND a.confidence_score >= 8
AND a.assay_type IN ('B', 'F')
```

### Find a target by gene symbol (when you don't know the ChEMBL ID)

```sql
SELECT DISTINCT td.chembl_id, td.pref_name, td.organism
FROM target_dictionary td
JOIN target_components tc   ON td.tid = tc.tid
JOIN component_synonyms csy ON tc.component_id = csy.component_id
WHERE csy.component_synonym = 'KRAS'
  AND td.organism = 'Homo sapiens'
  AND td.target_type = 'SINGLE PROTEIN'
```

## Filter cheat sheet (values you'll see in the data)

### `standard_relation`
- `=` — exact measurement (keep)
- `>` — "at least this weak" (censored, usually drop)
- `<` — "at most this potent" (censored, usually drop)
- `~` — approximate (judgment call, often drop)

### `confidence_score` (0–9, ChEMBL's confidence the assay actually hits the target)
- `9` — homologous single-protein target, directly assigned (highest)
- `8` — direct single-protein assignment
- `7` — homologous protein complex
- `5–6` — protein family or complex (less specific)
- `0–4` — non-molecular, indirect, or unassigned (usually filter out)

**For our project: keep ≥ 8.**

### `assay_type`
- `B` — Binding (biochemical, direct affinity)
- `F` — Functional (cellular, downstream readout)
- `A` — ADMET (absorption/metabolism/excretion/toxicity)
- `T` — Toxicity
- `P` — Physicochemical (solubility, logP, etc.)
- `U` — Unclassified

**For our project: keep `B` and `F`.**

### `standard_units` (common values)
- `nM`, `uM`, `μM`, `mM`, `M` — concentration (use these)
- `pM` — picomolar (very potent)
- `%` — percent inhibition (NOT a concentration; can't compute pIC50 from this)
- `ug.mL-1`, `mg.kg-1` — mass-based units (need MW to convert)

### `standard_type` (what was measured)
- `IC50` — concentration for 50% inhibition (functional potency)
- `Ki` — inhibition constant (binding affinity)
- `Kd` — dissociation constant (binding affinity)
- `EC50` — concentration for 50% effect (functional, usually agonist)
- `AC50`, `GI50` — similar variants
- `% Inhibition` — not a concentration, skip for regression

## Quick exploration commands

When you need to figure out what's in a table you haven't used:

```python
# List every table in ChEMBL
pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", con)

# Peek at the first 5 rows of any table
pd.read_sql("SELECT * FROM <table_name> LIMIT 5", con)

# Show every column in a table with its type
pd.read_sql("PRAGMA table_info(<table_name>)", con)

# Count rows in a table
pd.read_sql("SELECT COUNT(*) FROM <table_name>", con)

# Sample distinct values in a column (useful for categorical columns like assay_type)
pd.read_sql("SELECT DISTINCT <column> FROM <table_name> LIMIT 20", con)
```

## Mental shortcut

Writing a query is just:

1. **What do I want to end up with?** (one row per what?)
2. **Which table has that?** (start your FROM clause there)
3. **What other info do I need?** (each = one JOIN)
4. **Which rows do I want?** (each filter = one AND in WHERE)
5. **Which columns do I want to see?** (the SELECT list)

If you can describe the question in plain English, the SQL writes itself.
