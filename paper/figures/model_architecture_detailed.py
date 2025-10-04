#!/usr/bin/env python3
"""
Generate detailed model architecture diagrams for the research paper
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_detailed_architecture_diagram():
    """Create a detailed model architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12), dpi=300)
    
    # Define colors
    colors = {
        'input': '#E8F4FD',
        'attention': '#FFE6E6', 
        'fusion': '#E6F7E6',
        'dueling': '#FFF2E6',
        'output': '#F0E6FF'
    }
    
    # Input layer
    input_box = FancyBboxPatch((1, 9), 3, 1.5, boxstyle="round,pad=0.1", 
                              facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2.5, 9.75, 'Market Features\n(270 features)', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    input_box2 = FancyBboxPatch((1, 7), 3, 1.5, boxstyle="round,pad=0.1", 
                               facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(input_box2)
    ax.text(2.5, 7.75, 'Sentiment Features\n(90 features)', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    # Multi-Head Attention Module
    attention_box = FancyBboxPatch((6, 8), 4, 2, boxstyle="round,pad=0.1", 
                                  facecolor=colors['attention'], edgecolor='black', linewidth=2)
    ax.add_patch(attention_box)
    ax.text(8, 9, 'Multi-Head\nSelf-Attention', ha='center', va='center', 
           fontsize=11, fontweight='bold')
    
    # Attention heads detail
    for i in range(4):
        head_box = FancyBboxPatch((6.2 + i*0.8, 8.2), 0.6, 0.6, boxstyle="round,pad=0.05", 
                                 facecolor='white', edgecolor='gray', linewidth=1)
        ax.add_patch(head_box)
        ax.text(6.5 + i*0.8, 8.5, f'H{i+1}', ha='center', va='center', 
               fontsize=8, fontweight='bold')
    
    # Temporal Encoding
    encoding_box = FancyBboxPatch((6, 6), 4, 1.5, boxstyle="round,pad=0.1", 
                                 facecolor=colors['attention'], edgecolor='black', linewidth=2)
    ax.add_patch(encoding_box)
    ax.text(8, 6.75, 'Temporal\nPositional Encoding', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    # Cross-Attention Fusion
    fusion_box = FancyBboxPatch((12, 7.5), 4, 2, boxstyle="round,pad=0.1", 
                               facecolor=colors['fusion'], edgecolor='black', linewidth=2)
    ax.add_patch(fusion_box)
    ax.text(14, 8.5, 'Cross-Attention\nFusion Module', ha='center', va='center', 
           fontsize=11, fontweight='bold')
    
    # Dueling Network
    dueling_box = FancyBboxPatch((12, 5), 4, 2, boxstyle="round,pad=0.1", 
                                facecolor=colors['dueling'], edgecolor='black', linewidth=2)
    ax.add_patch(dueling_box)
    ax.text(14, 6, 'Dueling DQN\nNetwork', ha='center', va='center', 
           fontsize=11, fontweight='bold')
    
    # Value and Advantage streams
    value_box = FancyBboxPatch((12.2, 5.2), 1.6, 0.6, boxstyle="round,pad=0.05", 
                              facecolor='white', edgecolor='gray', linewidth=1)
    ax.add_patch(value_box)
    ax.text(13, 5.5, 'Value\nStream', ha='center', va='center', 
           fontsize=8, fontweight='bold')
    
    advantage_box = FancyBboxPatch((14.2, 5.2), 1.6, 0.6, boxstyle="round,pad=0.05", 
                                  facecolor='white', edgecolor='gray', linewidth=1)
    ax.add_patch(advantage_box)
    ax.text(15, 5.5, 'Advantage\nStream', ha='center', va='center', 
           fontsize=8, fontweight='bold')
    
    # Output layer
    output_box = FancyBboxPatch((12, 2.5), 4, 1.5, boxstyle="round,pad=0.1", 
                               facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(14, 3.25, 'Q-Values\n(9 stocks)', ha='center', va='center', 
           fontsize=11, fontweight='bold')
    
    # Arrows
    arrows = [
        ((2.5, 9), (6, 9)),      # Market to Attention
        ((2.5, 7), (6, 7.5)),    # Sentiment to Attention
        ((10, 9), (12, 8.5)),    # Attention to Fusion
        ((10, 6.75), (12, 7.5)), # Encoding to Fusion
        ((14, 7.5), (14, 7)),    # Fusion to Dueling
        ((14, 5), (14, 4)),      # Dueling to Output
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=20, fc="red", lw=2)
        ax.add_patch(arrow)
    
    # Add mathematical formulas
    ax.text(8, 4, 'Q(s,a) = V(s) + A(s,a) - (1/|A|)ΣA(s,a\')', 
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
    
    ax.text(8, 3, 'Attention(Q,K,V) = softmax(QK^T/√d_k)V', 
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
    
    ax.set_xlim(0, 18)
    ax.set_ylim(1, 11)
    ax.set_title('MHA-DQN Detailed Architecture', fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('paper/figures/detailed_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_training_flow_diagram():
    """Create training process flow diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10), dpi=300)
    
    # Define colors
    colors = {
        'data': '#E8F4FD',
        'model': '#FFE6E6',
        'training': '#E6F7E6',
        'evaluation': '#FFF2E6'
    }
    
    # Data Collection
    data_box = FancyBboxPatch((1, 8), 3, 1.5, boxstyle="round,pad=0.1", 
                             facecolor=colors['data'], edgecolor='black', linewidth=2)
    ax.add_patch(data_box)
    ax.text(2.5, 8.75, 'Market Data\nCollection', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    # Feature Engineering
    feature_box = FancyBboxPatch((1, 6), 3, 1.5, boxstyle="round,pad=0.1", 
                                facecolor=colors['data'], edgecolor='black', linewidth=2)
    ax.add_patch(feature_box)
    ax.text(2.5, 6.75, 'Feature\nEngineering', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    # Environment Setup
    env_box = FancyBboxPatch((1, 4), 3, 1.5, boxstyle="round,pad=0.1", 
                            facecolor=colors['data'], edgecolor='black', linewidth=2)
    ax.add_patch(env_box)
    ax.text(2.5, 4.75, 'Portfolio\nEnvironment', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    # Model Initialization
    model_box = FancyBboxPatch((6, 6), 3, 2, boxstyle="round,pad=0.1", 
                              facecolor=colors['model'], edgecolor='black', linewidth=2)
    ax.add_patch(model_box)
    ax.text(7.5, 7, 'MHA-DQN\nInitialization', ha='center', va='center', 
           fontsize=11, fontweight='bold')
    
    # Training Loop
    training_box = FancyBboxPatch((11, 7), 4, 2, boxstyle="round,pad=0.1", 
                                 facecolor=colors['training'], edgecolor='black', linewidth=2)
    ax.add_patch(training_box)
    ax.text(13, 8, 'Training Loop\n(100 episodes)', ha='center', va='center', 
           fontsize=11, fontweight='bold')
    
    # Experience Replay
    replay_box = FancyBboxPatch((11, 4.5), 4, 1.5, boxstyle="round,pad=0.1", 
                               facecolor=colors['training'], edgecolor='black', linewidth=2)
    ax.add_patch(replay_box)
    ax.text(13, 5.25, 'Experience\nReplay Buffer', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    # Model Update
    update_box = FancyBboxPatch((11, 2.5), 4, 1.5, boxstyle="round,pad=0.1", 
                               facecolor=colors['training'], edgecolor='black', linewidth=2)
    ax.add_patch(update_box)
    ax.text(13, 3.25, 'Model\nUpdate', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    # Evaluation
    eval_box = FancyBboxPatch((6, 2.5), 3, 2, boxstyle="round,pad=0.1", 
                             facecolor=colors['evaluation'], edgecolor='black', linewidth=2)
    ax.add_patch(eval_box)
    ax.text(7.5, 3.5, 'Model\nEvaluation', ha='center', va='center', 
           fontsize=11, fontweight='bold')
    
    # Arrows
    arrows = [
        ((2.5, 8), (2.5, 7.5)),    # Data to Features
        ((2.5, 6), (2.5, 5.5)),    # Features to Environment
        ((4, 4.75), (6, 7)),       # Environment to Model
        ((9, 7), (11, 8)),         # Model to Training
        ((13, 7), (13, 6)),        # Training to Replay
        ((13, 4.5), (13, 4)),      # Replay to Update
        ((11, 3.25), (9, 3.5)),    # Update to Evaluation
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=20, fc="blue", lw=2)
        ax.add_patch(arrow)
    
    # Add training details
    ax.text(7.5, 1.5, 'Training Details:\n• 100 episodes × 252 steps\n• Prioritized Experience Replay\n• Target Network Updates\n• ε-greedy Exploration', 
           ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
    
    ax.set_xlim(0, 16)
    ax.set_ylim(1, 10)
    ax.set_title('MHA-DQN Training Process Flow', fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('paper/figures/training_flow.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_attention_mechanism_diagram():
    """Create detailed attention mechanism diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=300)
    
    # Multi-head attention structure
    # Input
    input_box = FancyBboxPatch((1, 6), 2, 1, boxstyle="round,pad=0.1", 
                              facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2, 6.5, 'Input\nFeatures', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    # Linear transformations
    q_box = FancyBboxPatch((4, 7), 1.5, 0.8, boxstyle="round,pad=0.1", 
                          facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(q_box)
    ax.text(4.75, 7.4, 'Q', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    k_box = FancyBboxPatch((4, 6), 1.5, 0.8, boxstyle="round,pad=0.1", 
                          facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(k_box)
    ax.text(4.75, 6.4, 'K', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    v_box = FancyBboxPatch((4, 5), 1.5, 0.8, boxstyle="round,pad=0.1", 
                          facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(v_box)
    ax.text(4.75, 5.4, 'V', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    # Attention computation
    attention_box = FancyBboxPatch((7, 5.5), 2.5, 1.5, boxstyle="round,pad=0.1", 
                                  facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(attention_box)
    ax.text(8.25, 6.25, 'Attention\nComputation', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    # Multi-head concatenation
    concat_box = FancyBboxPatch((10.5, 6), 2, 1, boxstyle="round,pad=0.1", 
                               facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax.add_patch(concat_box)
    ax.text(11.5, 6.5, 'Concat\nHeads', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    # Output
    output_box = FancyBboxPatch((10.5, 4), 2, 1, boxstyle="round,pad=0.1", 
                               facecolor='lightpink', edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(11.5, 4.5, 'Output', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    # Arrows
    arrows = [
        ((3, 6.5), (4, 7.2)),     # Input to Q
        ((3, 6.5), (4, 6.4)),     # Input to K
        ((3, 6.5), (4, 5.2)),     # Input to V
        ((5.5, 7.4), (7, 6.8)),   # Q to Attention
        ((5.5, 6.4), (7, 6.2)),   # K to Attention
        ((5.5, 5.4), (7, 5.8)),   # V to Attention
        ((9.5, 6.25), (10.5, 6.5)), # Attention to Concat
        ((11.5, 6), (11.5, 5)),   # Concat to Output
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=15, fc="red", lw=1.5)
        ax.add_patch(arrow)
    
    # Add attention formula
    ax.text(8.25, 3, 'Attention(Q,K,V) = softmax(QK^T/√d_k)V', 
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
    
    # Add multi-head formula
    ax.text(8.25, 2, 'MultiHead(Q,K,V) = Concat(head₁,...,headₕ)W^O', 
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
    
    ax.set_xlim(0, 14)
    ax.set_ylim(1, 9)
    ax.set_title('Multi-Head Attention Mechanism', fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('paper/figures/attention_mechanism.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_comparison_diagram():
    """Create performance comparison diagram"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=300)
    
    # Performance metrics comparison
    methods = ['MHA-DQN\n(Ours)', 'Equal\nWeight', 'Mean-\nVariance', 'Risk\nParity', 'Standard\nDQN', 'Dueling\nDQN']
    sharpe_ratios = [1.265, 0.389, 0.571, 0.534, 0.678, 0.789]
    returns = [41.75, 17.49, 22.15, 19.87, 28.34, 31.22]
    
    # Sharpe ratio comparison
    bars1 = ax1.bar(methods, sharpe_ratios, color=['red', 'lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink'])
    ax1.set_title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Sharpe Ratio', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, sharpe_ratios):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Annual return comparison
    bars2 = ax2.bar(methods, returns, color=['red', 'lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink'])
    ax2.set_title('Annual Return Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Annual Return (%)', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars2, returns):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('paper/figures/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    import os
    os.makedirs('paper/figures', exist_ok=True)
    
    print("Generating detailed architecture diagram...")
    create_detailed_architecture_diagram()
    
    print("Generating training flow diagram...")
    create_training_flow_diagram()
    
    print("Generating attention mechanism diagram...")
    create_attention_mechanism_diagram()
    
    print("Generating performance comparison diagram...")
    create_performance_comparison_diagram()
    
    print("All diagrams generated successfully!")
