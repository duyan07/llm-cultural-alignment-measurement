# Variable Mapping: SPSS Syntax → CSV Reality

## CONFIRMED MAPPINGS

### Merge Key
- **SPSS**: `s007_01`
- **WVS CSV**: `S007_01`
- **EVS CSV**: `S007_01`
- **Type**: Unique respondent identifier

### Wave Variables
- **WVS Wave**:
  - SPSS: `S002`
  - CSV: `s002`
  - Values: 1-7 (we filter to 5, 6, 7)

- **EVS Wave**:
  - SPSS: `S002evs`
  - CSV: `S002EVS`
  - Values: 1-5 (we filter to 4, 5)

### Country-Wave
- **SPSS**: `S024`
- **CSV**: `S024` (both)
- **Coverage**: 100% (no missing values)

## Missing Data Notes

- Original SPSS codes (-5 to -1) → converted to `NaN` by R
- 60% NaN in WVS is NORMAL (longitudinal + country variation)
- 51% NaN in EVS is NORMAL (same reasons)