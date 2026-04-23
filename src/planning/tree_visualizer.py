import matplotlib.pyplot as plt
import networkx as nx
import os
from typing import List, Dict, Any

class TreeVisualizer:
    @staticmethod
    def draw_tree(tree_data: List[Dict], chosen_ids: List[int], filepath: str, agent_id: int, phase: str = "aggressive", reason: str = "SUCCESS"):
        """
        Generates a PNG image of the search tree.
        """
        if not tree_data:
            return

        G = nx.DiGraph()
        labels = {}
        color_map = []
        
        # 1. Build Graph
        # Root check
        root_added = False
        for node in tree_data:
            node_id = node["node_id"]
            parent_id = node.get("parent_id")
            
            # Label: Action + X position + F cost + Telemetry
            action = node.get("action", "START")
            lane = node.get("lane_type", "?")
            f_cost = node.get("f", 0.0)
            acc = node.get("acc", 0.0)
            steer = node.get("steer", 0.0)
            x_pos = node.get("state", [0,0,0,0,0])[0]
            fail_type = node.get("fail_type", "NONE")
            
            if not node.get("valid", True):
                label_text = f"{action}\nex:{x_pos:.1f}m\n{fail_type}\nF:{f_cost:.2f} [{lane}]"
            else:
                label_text = f"{action}\nex:{x_pos:.1f}m\nF:{f_cost:.2f} [{lane}]"
            
            labels[node_id] = label_text
            
            if parent_id is not None:
                G.add_edge(parent_id, node_id)
            else:
                G.add_node(node_id)
                root_added = True

        # 2. Assign Colors
        for node_id in G.nodes():
            # Find the node info in tree_data
            info = next((n for n in tree_data if n["node_id"] == node_id), None)
            
            if info is None: # Should not happen
                color_map.append("gray")
                continue
                
            if info.get("parent_id") is None:
                color_map.append("skyblue") # Root
            elif node_id in chosen_ids:
                color_map.append("limegreen") # Chosen path
            elif not info.get("valid", True):
                color_map.append("salmon") # Rejected/Invalid
            else:
                color_map.append("gold") # Valid candidate
        
        # 3. Layout (Hierarchy)
        try:
            # Simple hierarchical layout based on node_id (since they are added in order)
            # or use Graphviz if available (but it's not)
            # We'll use a multipartite layout simulation
            pos = {}
            # Group by "depth" using ID or a BFS
            depths = {}
            if root_added:
                root_id = next(n["node_id"] for n in tree_data if n.get("parent_id") is None)
                depths = nx.single_source_shortest_path_length(G, root_id)
            
            # Position by depth (Y) and spread (X)
            counts = {}
            for nid in G.nodes():
                d = depths.get(nid, 0)
                counts[d] = counts.get(d, 0) + 1
                # Increase spread significantly
                pos[nid] = (counts[d] * 3.5, -d * 6.0) 
                
            # 4. Plot
            plt.figure(figsize=(20, 15))
            nx.draw(G, pos, labels=labels, with_labels=True, node_color=color_map, 
                    node_size=4500, font_size=7, font_weight="bold", arrows=True, 
                    edge_color="gray", alpha=0.8, width=1.2)
            
            title_text = f"Agent {agent_id} | Phase: {phase.upper()} | Result: {reason}\nGreen=Selected | Gold=Explored | Red=Rejected/Collision"
            plt.title(title_text, fontsize=18, fontweight='bold', color='darkred' if "FAILED" in reason or "BLOCKED" in reason else 'darkgreen')
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            plt.savefig(filepath, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"!!! Error generating tree image for Agent {agent_id}: {e}")
