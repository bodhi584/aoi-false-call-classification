# From 75 features to 206 model inputs

The data set [4] has 75 features per flagged solder joint: 70 measurements and 5 categorical features.
The model cannot read a category number as a name, so **categorical features are turned into binary 0/1 columns, one per category**. Measurements stay as they are.

| Feature | Type | Distinct values | Model inputs |
|---|---|---|---|
| `inspection_feat…` (70 measurements) | numerical | – | **70** |
| `inspection_type` | categorical | 5 | 5 |
| `meta_feat1` | categorical | 75 | 75 |
| `meta_feat2` | categorical | 5 | 5 |
| `meta_feat3` | categorical | 2 | 2 |
| `meta_feat4` | categorical | 49 | 49 |
| **Total** | | | **70 + 136 = 206** |

## Example: `inspection_type`

Before (one column, the number is a name):

| Joint | inspection_type |
|---|---|
| A | 2 |
| B | 0 |

After (one 0/1 column per category):

| Joint | type 0 | type 1 | type 2 | type 3 | type 4 |
|---|---|---|---|---|---|
| A | 0 | 0 | 1 | 0 | 0 |
| B | 1 | 0 | 0 | 0 | 0 |

Why: type 4 is not "more" than type 2. Read as numbers, the model could treat the categories as amounts.
As 0/1 columns, it can ask directly: "is this joint of type 2?"

Counts: all 440,274 rows of the data set. The coding is done in [`scripts/replicate_pfab2025.py`](../scripts/replicate_pfab2025.py) (one-hot encoding, lines 58–59).
