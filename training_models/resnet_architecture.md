# ResNet (Residual Network) Architecture

## Model Configuration
- **Input Size**: 8 (current image features)
- **Hidden Size**: 64
- **Number of Blocks**: 3
- **Output Size**: 3 (vx, vy, vz velocities)
- **Dropout Rate**: 0.2
- **Skip Connections**: Identity mapping with residual learning

## Architecture Diagram

```mermaid
graph TD
    %% Input Processing
    A[Input Features<br/>8 coordinates<br/>StandardScaler] --> B[Input Projection<br/>Linear: 8 → 64]
    B --> C[ReLU Activation]
    C --> D[Dropout<br/>p=0.2]
    
    %% Residual Block 1
    D --> E[Residual Block 1]
    subgraph ResBlock1 [Residual Block 1]
        E1[Linear: 64 → 64] --> E2[ReLU]
        E2 --> E3[Dropout p=0.2]
        E3 --> E4[Linear: 64 → 64]
        E4 --> E5[ReLU]
        E5 --> E6[Dropout p=0.2]
    end
    D --> E7[Skip Connection]
    E6 --> E8[Add: x + residual]
    E7 --> E8
    
    %% Residual Block 2
    E8 --> F[Residual Block 2]
    subgraph ResBlock2 [Residual Block 2]
        F1[Linear: 64 → 64] --> F2[ReLU]
        F2 --> F3[Dropout p=0.2]
        F3 --> F4[Linear: 64 → 64]
        F4 --> F5[ReLU]
        F5 --> F6[Dropout p=0.2]
    end
    E8 --> F7[Skip Connection]
    F6 --> F8[Add: x + residual]
    F7 --> F8
    
    %% Residual Block 3
    F8 --> G[Residual Block 3]
    subgraph ResBlock3 [Residual Block 3]
        G1[Linear: 64 → 64] --> G2[ReLU]
        G2 --> G3[Dropout p=0.2]
        G3 --> G4[Linear: 64 → 64]
        G4 --> G5[ReLU]
        G5 --> G6[Dropout p=0.2]
    end
    F8 --> G7[Skip Connection]
    G6 --> G8[Add: x + residual]
    G7 --> G8
    
    %% Output Projection
    G8 --> H[Output Projection<br/>Linear: 64 → 32]
    H --> I[ReLU Activation]
    I --> J[Dropout<br/>p=0.2]
    J --> K[Final Layer<br/>Linear: 32 → 3]
    K --> L[Output<br/>vx, vy, vz<br/>StandardScaler⁻¹]
    
    %% Styling
    classDef input fill:#e1f5fe
    classDef projection fill:#f3e5f5
    classDef activation fill:#fff3e0
    classDef dropout fill:#ffebee
    classDef residual fill:#e8f5e8
    classDef skip fill:#fff9c4
    classDef add fill:#f1f8e9
    classDef output fill:#e0f2f1
    
    class A input
    class B,H,K projection
    class C,I activation
    class D,J dropout
    class E1,E4,F1,F4,G1,G4 projection
    class E2,E5,F2,F5,G2,G5 activation
    class E3,E6,F3,F6,G3,G6 dropout
    class E7,F7,G7 skip
    class E8,F8,G8 add
    class L output
```

## Layer Details

### Input Processing
- **StandardScaler**: Normalizes input features to zero mean, unit variance
- **Input Shape**: (batch_size, 8)
- **Input Projection**: Transforms 8 features to 64-dimensional hidden space

### Residual Blocks (×3)
Each residual block contains:
- **Linear Layer 1**: 64 → 64 neurons
- **ReLU Activation**: Non-linear activation function
- **Dropout**: 20% of neurons randomly set to zero during training
- **Linear Layer 2**: 64 → 64 neurons
- **ReLU Activation**: Non-linear activation function
- **Dropout**: 20% of neurons randomly set to zero during training
- **Skip Connection**: Identity mapping `F(x) = H(x) + x`

### Residual Learning Formula
```
Output = x + F(x)
```
Where:
- `x` is the input to the residual block
- `F(x)` is the residual function learned by the block
- The network learns the residual mapping rather than the direct mapping

### Skip Connection Benefits
- **Gradient Flow**: Enables direct gradient flow through skip connections
- **Deep Training**: Allows training of deeper networks without vanishing gradients
- **Identity Mapping**: Preserves information from earlier layers
- **Easier Optimization**: Network can learn identity function if needed

### Output Processing
- **Output Projection**: 64 → 32 → 3 neurons
- **Final Activation**: ReLU before final layer
- **Final Dropout**: 20% regularization before output
- **Linear Output**: No activation on final layer for regression

### Output Processing
- **Inverse StandardScaler**: Denormalizes predictions back to original scale
- **Output Shape**: (batch_size, 3)

## Mathematical Formulation

### Residual Block Function
```
H(x) = F(x) + x
```

### Complete Forward Pass
```
x₀ = InputProjection(input)
x₁ = ResidualBlock₁(x₀) = F₁(x₀) + x₀
x₂ = ResidualBlock₂(x₁) = F₂(x₁) + x₁
x₃ = ResidualBlock₃(x₂) = F₃(x₂) + x₂
output = OutputProjection(x₃)
```

## Training Details
- **Loss Function**: MSE (Mean Squared Error)
- **Optimizer**: Adam with weight decay (1e-3)
- **Learning Rate**: 0.0005 with ReduceLROnPlateau scheduler
- **Batch Size**: 64
- **Early Stopping**: Patience 15 epochs, min_delta 1e-5
- **Gradient Flow**: Skip connections enable stable gradient propagation 