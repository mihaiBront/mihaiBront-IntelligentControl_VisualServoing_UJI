# FNN (Feedforward Neural Network) Architecture

## Model Configuration
- **Input Size**: 8 (current image features)
- **Hidden Layers**: [64, 32]
- **Output Size**: 3 (vx, vy, vz velocities)
- **Dropout Rate**: 0.3
- **Batch Normalization**: Enabled
- **Activation**: ReLU
- **Weight Initialization**: He (Kaiming) for hidden layers, Xavier for output

## Architecture Diagram

```mermaid
graph TD
    %% Input Layer
    A[Input Features<br/>8 coordinates<br/>StandardScaler] --> B[Linear Layer 1<br/>8 → 64<br/>He Init]
    
    %% First Hidden Layer
    B --> C[BatchNorm1d<br/>64 features]
    C --> D[ReLU Activation]
    D --> E[Dropout<br/>p=0.3]
    
    %% Second Hidden Layer
    E --> F[Linear Layer 2<br/>64 → 32<br/>He Init]
    F --> G[BatchNorm1d<br/>32 features]
    G --> H[ReLU Activation]
    H --> I[Dropout<br/>p=0.3]
    
    %% Output Layer
    I --> J[Output Layer<br/>32 → 3<br/>Xavier Init<br/>gain=0.1]
    J --> K[Output<br/>vx, vy, vz<br/>StandardScaler⁻¹]
    
    %% Styling
    classDef input fill:#e1f5fe
    classDef linear fill:#f3e5f5
    classDef norm fill:#e8f5e8
    classDef activation fill:#fff3e0
    classDef dropout fill:#ffebee
    classDef output fill:#e0f2f1
    
    class A input
    class B,F,J linear
    class C,G norm
    class D,H activation
    class E,I dropout
    class K output
```

## Layer Details

### Input Processing
- **StandardScaler**: Normalizes input features to zero mean, unit variance
- **Input Shape**: (batch_size, 8)

### Hidden Layer 1
- **Linear**: 8 → 64 neurons
- **BatchNorm1d**: Normalizes activations across batch dimension
- **ReLU**: Non-linear activation function
- **Dropout**: 30% of neurons randomly set to zero during training

### Hidden Layer 2
- **Linear**: 64 → 32 neurons
- **BatchNorm1d**: Normalizes activations across batch dimension
- **ReLU**: Non-linear activation function
- **Dropout**: 30% of neurons randomly set to zero during training

### Output Layer
- **Linear**: 32 → 3 neurons (vx, vy, vz)
- **Xavier Initialization**: Smaller weights (gain=0.1) for stability
- **No Activation**: Linear output for regression

### Output Processing
- **Inverse StandardScaler**: Denormalizes predictions back to original scale
- **Output Shape**: (batch_size, 3)

## Training Details
- **Loss Function**: MSE (Mean Squared Error)
- **Optimizer**: Adam with weight decay (1e-3)
- **Learning Rate**: 0.0005 with ReduceLROnPlateau scheduler
- **Batch Size**: 64
- **Early Stopping**: Patience 15 epochs, min_delta 1e-5 