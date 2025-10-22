#!/usr/bin/env python3
"""
Render NetworkX graph with US state boundaries to SVG.

Takes a pickled graph file and renders state polygons as white shapes
with dark grey borders on a transparent background.
"""

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from shapely.wkt import loads as wkt_loads


def parse_wkt_polygon(wkt):
    """Parse WKT polygon string and return matplotlib patches."""
    try:
        geom = wkt_loads(wkt)
        polygons = []

        if geom.geom_type == 'Polygon':
            polygons = [geom]
        elif geom.geom_type == 'MultiPolygon':
            polygons = list(geom.geoms)

        patches_list = []
        for poly in polygons:
            # Get exterior coordinates
            exterior_coords = list(poly.exterior.coords)
            if exterior_coords:
                patch = patches.Polygon(exterior_coords, closed=True)
                patches_list.append(patch)

            # Handle holes
            for interior in poly.interiors:
                hole_coords = list(interior.coords)
                if hole_coords:
                    # Holes are handled by matplotlib automatically with path operations
                    pass

        return patches_list
    except Exception as e:
        print(f"Error parsing WKT: {e}")
        return []


def calculate_label_bbox(text, fontsize=10):
    """Estimate bounding box dimensions for a text label."""
    # More accurate approximation for bold text
    char_width = fontsize * 0.8  # pixels per character (bold text is wider)
    char_height = fontsize * 1.4  # line height with padding

    width = len(text) * char_width
    height = char_height

    return width, height


def force_directed_label_layout(labels, iterations=100, k_attract=0.005, k_repel=15000):
    """
    Apply force-directed algorithm to separate overlapping labels.

    Args:
        labels: List of dicts with 'text', 'orig_x', 'orig_y', 'x', 'y', 'width', 'height'
        iterations: Number of force-directed iterations
        k_attract: Attraction force constant (toward original position)
        k_repel: Repulsion force constant (away from overlapping labels)
    """
    positions = np.array([[label['x'], label['y']] for label in labels])
    orig_positions = np.array([[label['orig_x'], label['orig_y']] for label in labels])

    for iteration in range(iterations):
        forces = np.zeros_like(positions)

        # Attraction force toward original centroid positions
        attract_force = k_attract * (orig_positions - positions)
        forces += attract_force

        # Repulsion forces between overlapping labels
        for i, label_i in enumerate(labels):
            for j, label_j in enumerate(labels):
                if i >= j:
                    continue

                # Check if bounding boxes overlap
                dx = positions[j][0] - positions[i][0]
                dy = positions[j][1] - positions[i][1]

                # Minimum separation needed (half-widths + half-heights)
                min_dx = (label_i['width'] + label_j['width']) / 2
                min_dy = (label_i['height'] + label_j['height']) / 2

                # Only apply repulsion if bounding boxes actually overlap
                if abs(dx) < min_dx and abs(dy) < min_dy:
                    # Calculate actual overlap amounts (positive values mean overlap)
                    overlap_x = min_dx - abs(dx)
                    overlap_y = min_dy - abs(dy)

                    # Only repel if there's actual overlap in both dimensions
                    if overlap_x > 0 and overlap_y > 0:
                        # Repulsion strength based on overlap area
                        overlap_area = overlap_x * overlap_y
                        repel_strength = k_repel * overlap_area

                        # Direction of repulsion (away from each other)
                        if abs(dx) < 1:
                            dx = 1 if dx >= 0 else -1  # Avoid division by zero
                        if abs(dy) < 1:
                            dy = 1 if dy >= 0 else -1

                        distance = max(np.sqrt(dx**2 + dy**2), 1)  # Prevent division by zero
                        unit_dx = dx / distance
                        unit_dy = dy / distance

                        # Apply repulsive forces (push apart)
                        forces[i][0] -= repel_strength * unit_dx
                        forces[i][1] -= repel_strength * unit_dy
                        forces[j][0] += repel_strength * unit_dx
                        forces[j][1] += repel_strength * unit_dy

        # Apply forces with damping
        damping = 0.3
        positions += forces * damping * 0.0005  # Scale factor for movement

    # Update label positions with bounds checking
    for i, label in enumerate(labels):
        # Limit how far labels can move from their original positions
        max_movement = 300000  # 300km max movement in projected units

        dx = positions[i][0] - label['orig_x']
        dy = positions[i][1] - label['orig_y']
        distance = np.sqrt(dx**2 + dy**2)

        if distance > max_movement:
            # Scale back to maximum allowed movement
            scale = max_movement / distance
            dx *= scale
            dy *= scale

        label['x'] = label['orig_x'] + dx
        label['y'] = label['orig_y'] + dy


def render_graph(graph_file, output_file):
    """Render the pickled graph to SVG."""
    # Load the graph
    print(f"Loading graph from {graph_file}...")
    with open(graph_file, 'rb') as f:
        G = pickle.load(f)

    print(f"Loaded graph with {G.number_of_nodes()} nodes")

    # Set up the plot
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_aspect('equal')

    # Remove axes and make background transparent
    ax.set_axis_off()
    fig.patch.set_alpha(0.0)

    all_patches = []

    # Process each state polygon
    for state_code, data in G.nodes(data=True):
        wkt = data.get('wkt')
        if wkt:
            state_patches = parse_wkt_polygon(wkt)
            all_patches.extend(state_patches)

    print(f"Created {len(all_patches)} polygon patches")

    # Create patch collection
    if all_patches:
        collection = PatchCollection(all_patches,
                                   facecolor='white',
                                   edgecolor='#333333',  # dark grey
                                   linewidth=1.0,
                                   alpha=1.0)
        ax.add_collection(collection)

        # Set plot limits based on patch extents
        ax.autoscale()

    # Get map bounds for scaling calculations
    map_bounds = ax.get_xlim(), ax.get_ylim()
    map_width = map_bounds[0][1] - map_bounds[0][0]

    # Prepare labels for force-directed positioning
    labels = []
    fontsize = 10
    for state_code, data in G.nodes(data=True):
        x = data.get('x')
        y = data.get('y')
        seats = data.get('seats', 0)
        if x is not None and y is not None:
            text = f"{state_code} ({seats})" if seats > 0 else state_code
            width, height = calculate_label_bbox(text, fontsize)

            # Calculate label dimensions based on map scale
            # Estimate: label should be proportional to map width
            char_scale = map_width / 200  # Extremely large scale factor
            data_width = len(text) * char_scale * 1.2  # Character width
            data_height = char_scale * 3.0  # Line height

            labels.append({
                'text': text,
                'orig_x': x,
                'orig_y': y,
                'x': x,
                'y': y,
                'width': data_width,
                'height': data_height
            })

    print(f"Applying force-directed layout to {len(labels)} labels...")

    # Apply force-directed algorithm to separate overlapping labels
    force_directed_label_layout(labels, iterations=150)

    # Draw labels at their adjusted positions
    for label in labels:
        # Draw the text label
        ax.text(label['x'], label['y'], label['text'], ha='center', va='center',
               fontsize=fontsize, fontweight='bold', color='black')

    # Reset plot limits to original bounds (before labels were added)
    if all_patches:
        ax.autoscale()

    # Save as SVG
    print(f"Saving to {output_file}...")
    plt.savefig(output_file, format='svg', transparent=True,
                bbox_inches='tight', pad_inches=0.1)
    plt.close()

    print(f"SVG saved successfully to {output_file}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python render-graph.py <graph.pickle>")
        sys.exit(1)

    graph_file = sys.argv[1]
    output_file = "us-states.svg"

    render_graph(graph_file, output_file)


if __name__ == "__main__":
    main()