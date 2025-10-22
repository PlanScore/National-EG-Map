#!/usr/bin/env python3
"""
Render NetworkX graph with US state boundaries to SVG.

Takes a pickled graph file and renders state polygons as white shapes
with dark grey borders on a transparent background.
"""

import sys
import pickle
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

    # Add state labels at center coordinates
    for state_code, data in G.nodes(data=True):
        x = data.get('x')
        y = data.get('y')
        if x is not None and y is not None:
            ax.text(x, y, state_code, ha='center', va='center',
                   fontsize=10, fontweight='bold', color='black')

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