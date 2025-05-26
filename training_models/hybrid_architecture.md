# Hybrid CNN-LSTM Architecture

## Model Configuration
- **Input Size**: 8 (current image features)
- **CNN Channels**: [16, 32]
- **LSTM Hidden Size**: 64
- **LSTM Layers**: 2
- **Output Size**: 3 (vx, vy, vz velocities)
- **Dropout Rate**: 0.2
- **Sequence Length**: 10

## Architecture Diagram

```mermaid
graph TD
    %% Input Processing
    A[Input Sequence<br/>Shape: (batch, 10, 8)<br/>StandardScaler] --> B[Reshape for CNN<br/>(batch×10, 1, 8)]
    
    %% CNN Feature Extraction
    B --> C[Conv1d Layer 1<br/>1 → 16 channels<br/>kernel=3, padding=1]
    C --> D[ReLU Activation]
    D --> E[MaxPool1d<br/>kernel=2, stride=2]
    E --> F[Dropout<br/>p=0.2]
    
    F --> G[Conv1d Layer 2<br/>16 → 32 channels<br/>kernel=3, padding=1]
    G --> H[ReLU Activation]
    H --> I[MaxPool1d<br/>kernel=2, stride=2]
    I --> J[Dropout<br/>p=0.2]
    
    %% Reshape for LSTM
    J --> K[Reshape for LSTM<br/>(batch, 10, features)]
    
    %% LSTM Processing
    K --> L[LSTM Layer 1<br/>Input: CNN_features<br/>Hidden: 64]
    L --> M[LSTM Layer 2<br/>Hidden: 64<br/>Dropout: 0.2<br/>between layers]
    
    %% Sequence Output
    M --> N[LSTM Output<br/>Shape: (batch, 10, 64)<br/>All timesteps]
    N --> O[Last Timestep<br/>Shape: (batch, 64)<br/>lstm_out[:, -1, :]]
    
    %% Fully Connected Head
    O --> P[Linear Layer 1<br/>64 → 32]
    P --> Q[ReLU Activation]
    Q --> R[Dropout<br/>p=0.2]
    R --> S[Linear Layer 2<br/>32 → 3]
    S --> T[Output<br/>vx, vy, vz<br/>StandardScaler⁻¹]
    
    %% CNN Feature Map Visualization
    subgraph CNN_Details [CNN Feature Extraction Details]
        U[Input: 8 features<br/>After Conv1+Pool: 4 features<br/>After Conv2+Pool: 2 features]
        V[Channel Evolution<br/>1 → 16 → 32 channels]
        W[Receptive Field<br/>Captures local patterns<br/>in feature space]
    end
    
    %% LSTM Memory Visualization
    subgraph LSTM_Details [LSTM Temporal Processing]
        X[Temporal Dependencies<br/>Processes 10 timesteps<br/>sequentially]
        Y[Memory Gates<br/>Input, Forget, Output<br/>gates control information]
        Z[Hidden State<br/>Carries information<br/>across timesteps]
    end
    
    %% Styling
    classDef input fill:#e1f5fe
    classDef reshape fill:#fff3e0
    classDef conv fill:#f3e5f5
    classDef activation fill:#fff3e0
    classDef pool fill:#e8f5e8
    classDef dropout fill:#ffebee
    classDef lstm fill:#f3e5f5
    classDef selection fill:#fff3e0
    classDef linear fill:#e8f5e8
    classDef output fill:#e0f2f1
    classDef details fill:#fce4ec
    
    class A input
    class B,K reshape
    class C,G conv
    class D,H,Q activation
    class E,I pool
    class F,J,R dropout
    class L,M lstm
    class N,O selection
    class P,S linear
    class T output
    class U,V,W,X,Y,Z details
```

## Layer Details

### Input Processing
- **StandardScaler**: Normalizes input features to zero mean, unit variance
- **Input Shape**: (batch_size, sequence_length=10, features=8)
- **Reshape**: (batch×10, 1, 8) for CNN processing

### CNN Feature Extraction
- **Conv1d Layer 1**: 1 → 16 channels, kernel_size=3, padding=1
- **MaxPool1d**: Reduces spatial dimension by factor of 2
- **Conv1d Layer 2**: 16 → 32 channels, kernel_size=3, padding=1
- **MaxPool1d**: Further reduces spatial dimension by factor of 2
- **Dropout**: 20% applied after each CNN block

### Spatial Dimension Reduction
```
Input: 8 features
After Conv1+Pool: 8/2 = 4 features
After Conv2+Pool: 4/2 = 2 features
Final CNN output: 2 × 32 = 64 features per timestep
```

### Reshape for LSTM
- **Shape Transformation**: (batch×10, 32, 2) → (batch, 10, 64)
- **Feature Flattening**: CNN spatial features flattened for temporal processing

### LSTM Temporal Processing
- **Layer 1**: CNN_features → Hidden size 64
- **Layer 2**: Hidden size 64 → Hidden size 64
- **Dropout**: 20% applied between LSTM layers
- **Sequence Processing**: Processes all 10 timesteps sequentially
- **Output Selection**: Only last timestep output used

### Fully Connected Head
- **Linear 1**: 64 → 32 neurons
- **ReLU**: Non-linear activation
- **Dropout**: 20% regularization
- **Linear 2**: 32 → 3 neurons (vx, vy, vz)

### Output Processing
- **Inverse StandardScaler**: Denormalizes predictions back to original scale
- **Output Shape**: (batch_size, 3)

## Hybrid Architecture Benefits

### CNN Component
- **Local Pattern Detection**: Captures spatial relationships in features
- **Translation Invariance**: Robust to small shifts in feature positions
- **Feature Hierarchy**: Learns increasingly complex spatial patterns
- **Dimensionality Reduction**: Reduces computational load for LSTM

### LSTM Component
- **Temporal Dependencies**: Models sequential relationships across timesteps
- **Long-term Memory**: Remembers important information over time
- **Variable Length Sequences**: Can handle different sequence lengths
- **Gradient Flow**: Maintains gradients across long sequences

### Combined Advantages
- **Spatial-Temporal Modeling**: Captures both spatial and temporal patterns
- **Feature Extraction**: CNN extracts relevant spatial features for LSTM
- **Reduced Complexity**: CNN preprocessing reduces LSTM input dimensionality
- **Complementary Strengths**: Combines CNN's spatial awareness with LSTM's temporal modeling

## Mathematical Formulation

### CNN Forward Pass
```
x₁ = MaxPool(ReLU(Conv1d(x₀)))
x₂ = MaxPool(ReLU(Conv1d(x₁)))
cnn_features = Flatten(x₂)
```

### LSTM Forward Pass
```
h₁, c₁ = LSTM₁(cnn_features)
h₂, c₂ = LSTM₂(h₁, c₁)
output = FC(h₂[:, -1, :])
```

## Training Details
- **Loss Function**: MSE (Mean Squared Error)
- **Optimizer**: Adam with weight decay (1e-3)
- **Learning Rate**: 0.0005 with ReduceLROnPlateau scheduler
- **Batch Size**: 64
- **Early Stopping**: Patience 15 epochs, min_delta 1e-5
- **Sequence Handling**: Sliding window approach for temporal dependencies 