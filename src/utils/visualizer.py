import matplotlib.pyplot as plt

def plot_comparison(history, gt_future, pred_future, roadgraph, is_controlled=None, interactions=None, title=None):
    """
    Plots a comparison of Ground Truth and Predicted trajectories.
    Enhanced aesthetics based on Waymax visualization standards.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 1. Plot Roadgraph (Map lines and lane boundaries)
    valid_rg = roadgraph[roadgraph[:, 7] > 0.5]
    if len(valid_rg) > 0:
        ax.scatter(valid_rg[:, 0], valid_rg[:, 1], s=5, color='#BDC3C7', alpha=0.4, label='Road / Lanes')

    # 2. Plot Interaction Agents (Surrounding Vehicles)
    if interactions is not None:
        # xyz at 0,1,2. Valid at 9. 
        valid_inter = interactions[interactions[:, 0, 9] > 0.5]
        if len(valid_inter) > 0:
            ax.scatter(valid_inter[:, 0, 0], valid_inter[:, 0, 1], s=60, marker='s', 
                       color='#ECF0F1', edgecolor='#7F8C8D', alpha=0.7, label='Interactions')

    # 3. Plot Target Agents
    num_agents = history.shape[0]
    for i in range(num_agents):
        controlled = is_controlled[i] if is_controlled is not None else False
        color_gt = '#F1C40F' if controlled else '#2ECC71'
        color_pred = '#E67E22' if controlled else '#E74C3C'
        label_prefix = f"Agent {i} (Controlled)" if controlled else f"Agent {i}"
        
        ax.plot(history[i, :, 0], history[i, :, 1], 'k.', markersize=4, alpha=0.3, label=f'{label_prefix} Past' if controlled else "")
        ax.plot(gt_future[i, :, 0], gt_future[i, :, 1], color=color_gt, 
                marker='o', markersize=4, linewidth=2.5, alpha=0.8, label=f'{label_prefix} GT')
        ax.plot(pred_future[i, :, 0], pred_future[i, :, 1], color=color_pred, 
                marker='x', markersize=5, linewidth=2, linestyle='--', alpha=0.8, label=f'{label_prefix} Pred')

    # 4. Set Limits and Aspect
    ax.set_aspect('equal')
    ax.set_xlim(-80, 80)
    ax.set_ylim(-80, 80)
    
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.set_facecolor('#FDFEFE')
    
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', color='#2C3E50')
    
    ax.legend(loc='upper right', fontsize='small', framealpha=0.6)
    plt.tight_layout()
    return fig
