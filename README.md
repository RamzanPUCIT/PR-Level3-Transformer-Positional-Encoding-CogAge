# Level 3 Transformer Attention with Positional Encoding on CogAge Time-Series Data

This project implements Level 3 Transformer scaled dot-product attention with positional encoding on CogAge time-series sensor data.

## Dataset

- Dataset: CogAge
- Selected activity/sample: Adeel_Bending_1
- Sensor: Phone Accelerometer
- Input features: x-axis, y-axis, z-axis
- Input shape: `(1007, 3)`

## Methodology

The timestamp column was removed and only sensor values were used.  
The input was normalized before applying attention.

### Step 1: Normalization

`X = (X_raw - mean) / std`

### Step 2: Positional Encoding

`X_pos = X + PE`

Positional encoding was added so that the Transformer can use the order of time-points.

### Step 3: Query, Key, Value

`Q = X_pos W_Q`  
`K = X_pos W_K`  
`V = X_pos W_V`

### Step 4: Scaled Dot-Product Attention

`Attention(Q,K,V) = softmax((QK^T) / sqrt(d_k)) V`

## Final Shapes

| Component | Shape |
|---|---|
| X | `(1007, 3)` |
| PE | `(1007, 3)` |
| X_pos | `(1007, 3)` |
| Q | `(1007, 8)` |
| K | `(1007, 8)` |
| V | `(1007, 8)` |
| Attention Scores | `(1007, 1007)` |
| Attention Weights | `(1007, 1007)` |
| Final Output | `(1007, 8)` |

## Visualizations

The project includes:

- Attention heatmap with positional encoding
- Positional encoding value plot
- Comparison of original input and position-aware input

## Key Learning

This project shows how positional encoding helps Transformer attention understand the temporal order of sensor time-series data.
