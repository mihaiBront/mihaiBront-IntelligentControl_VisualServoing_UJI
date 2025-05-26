# LSTM (Long Short-Term Memory) Architecture

## Model Configuration
- **Input Size**: 8 (current image features)
- **Hidden Size**: 64
- **Number of Layers**: 2
- **Output Size**: 3 (vx, vy, vz velocities)
- **Dropout Rate**: 0.2
- **Sequence Length**: 10
- **Batch First**: True

## Architecture Diagram

```mermaid
graph TD
    %% Input Sequence
    A[Input Sequence<br/>Shape: (batch, 10, 8)<br/>StandardScaler] --> B[LSTM Layer 1<br/>Input: 8<br/>Hidden: 64<br/>Bidirectional: False]
    
    %% LSTM Layers
    B --> C[LSTM Layer 2<br/>Hidden: 64<br/>Dropout: 0.2<br/>between layers]
    
    %% Sequence Output
    C --> D[LSTM Output<br/>Shape: (batch, 10, 64)<br/>All timesteps]
    
    %% Last Timestep Selection
    D --> E[Last Timestep<br/>Shape: (batch, 64)<br/>lstm_out[:, -1, :]]
    
    %% Fully Connected Layers
    E --> F[Linear Layer 1<br/>64 → 32]
    F --> G[ReLU Activation]
    G --> H[Dropout<br/>p=0.2]
    H --> I[Linear Layer 2<br/>32 → 3]
    I --> J[Output<br/>vx, vy, vz<br/>StandardScaler⁻¹]
    
    %% Memory Cell Visualization
    subgraph LSTM_Cell [LSTM Cell Details]
        K[Input Gate<br/>σ(Wᵢ·[hₜ₋₁,xₜ] + bᵢ)]
        L[Forget Gate<br/>σ(Wf·[hₜ₋₁,xₜ] + bf)]
        M[Output Gate<br/>σ(Wₒ·[hₜ₋₁,xₜ] + bₒ)]
        N[Cell State<br/>Cₜ = fₜ * Cₜ₋₁ + iₜ * C̃ₜ]
        O[Hidden State<br/>hₜ = oₜ * tanh(Cₜ)]
    end
    
    %% Styling
    classDef input fill:#e1f5fe
    classDef lstm fill:#f3e5f5
    classDef selection fill:#fff3e0
    classDef linear fill:#e8f5e8
    classDef activation fill:#fff3e0
    classDef dropout fill:#ffebee
    classDef output fill:#e0f2f1
    classDef memory fill:#fce4ec
    
    class A input
    class B,C lstm
    class D,E selection
    class F,I linear
    class G activation
    class H dropout
    class J output
    class K,L,M,N,O memory
```

## Layer Details

### Input Processing
- **StandardScaler**: Normalizes input features to zero mean, unit variance
- **Input Shape**: (batch_size, sequence_length=10, features=8)
- **Sequence**: 10 consecutive timesteps of 8 image features each

### LSTM Layers
- **Layer 1**: Input size 8 → Hidden size 64
- **Layer 2**: Hidden size 64 → Hidden size 64
- **Dropout**: 20% applied between LSTM layers (when num_layers > 1)
- **Batch First**: Input/output tensors have batch dimension first
- **Bidirectional**: False (unidirectional processing)

### LSTM Cell Components
- **Input Gate**: Controls what new information to store in cell state
- **Forget Gate**: Controls what information to discard from cell state
- **Output Gate**: Controls what parts of cell state to output as hidden state
- **Cell State**: Long-term memory component
- **Hidden State**: Short-term memory component passed to next timestep

### Sequence Processing
- **Output Selection**: Only the last timestep output is used: `lstm_out[:, -1, :]`
- **Shape Transformation**: (batch, 10, 64) → (batch, 64)

### Fully Connected Head
- **Linear 1**: 64 → 32 neurons
- **ReLU**: Non-linear activation
- **Dropout**: 20% of neurons randomly set to zero during training
- **Linear 2**: 32 → 3 neurons (vx, vy, vz)

### Output Processing
- **Inverse StandardScaler**: Denormalizes predictions back to original scale
- **Output Shape**: (batch_size, 3)

## Training Details
- **Loss Function**: MSE (Mean Squared Error)
- **Optimizer**: Adam with weight decay (1e-3)
- **Learning Rate**: 0.0005 with ReduceLROnPlateau scheduler
- **Batch Size**: 64
- **Early Stopping**: Patience 15 epochs, min_delta 1e-5
- **Sequence Handling**: Sliding window approach for temporal dependencies 