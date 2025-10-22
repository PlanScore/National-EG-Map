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

        if geom.geom_type == "Polygon":
            polygons = [geom]
        elif geom.geom_type == "MultiPolygon":
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


def bboxes_overlap(label1, label2):
    """Check if two label bounding boxes overlap."""
    # Get bbox corners
    left1 = label1["x"] - label1["width"] / 2
    right1 = label1["x"] + label1["width"] / 2
    top1 = label1["y"] + label1["height"] / 2
    bottom1 = label1["y"] - label1["height"] / 2

    left2 = label2["x"] - label2["width"] / 2
    right2 = label2["x"] + label2["width"] / 2
    top2 = label2["y"] + label2["height"] / 2
    bottom2 = label2["y"] - label2["height"] / 2

    # Check if they don't overlap (easier to check)
    if right1 <= left2 or right2 <= left1 or top1 <= bottom2 or top2 <= bottom1:
        return False
    return True


def has_any_overlaps(labels):
    """Check if any labels have overlapping bounding boxes."""
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if bboxes_overlap(labels[i], labels[j]):
                return True
    return False


def force_directed_label_layout(
    labels, max_iterations=300, k_attract=0.005, k_repel=50000
):
    """
    Apply force-directed algorithm to separate overlapping labels.
    Stops when no overlaps are detected or max iterations reached.

    Args:
        labels: List of dicts with 'text', 'orig_x', 'orig_y', 'x', 'y', 'width', 'height'
        max_iterations: Maximum number of iterations
        k_attract: Attraction force constant (toward original position)
        k_repel: Repulsion force constant (away from overlapping labels)
    """
    positions = np.array([[label["x"], label["y"]] for label in labels])
    # orig_positions = np.array([[label['orig_x'], label['orig_y']] for label in labels])  # Unused while attraction disabled

    for iteration in range(max_iterations):
        forces = np.zeros_like(positions)

        # Temporarily disable attraction force to debug repulsion
        # attract_force = k_attract * (orig_positions - positions)
        # forces += attract_force

        # Update label positions for accurate overlap checking
        for i, label in enumerate(labels):
            label["x"], label["y"] = positions[i]

        # Count overlaps for debugging
        overlaps_found = False
        overlap_count = 0

        # Repulsion forces between overlapping labels
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                label_i = labels[i]
                label_j = labels[j]

                # Check if bounding boxes overlap
                if bboxes_overlap(label_i, label_j):
                    overlaps_found = True
                    overlap_count += 1

                    # Calculate repulsion force
                    dx = positions[j][0] - positions[i][0]
                    dy = positions[j][1] - positions[i][1]

                    # Avoid division by zero
                    if abs(dx) < 1:
                        dx = 1 if dx >= 0 else -1
                    if abs(dy) < 1:
                        dy = 1 if dy >= 0 else -1

                    distance = max(np.sqrt(dx**2 + dy**2), 1)
                    unit_dx = dx / distance
                    unit_dy = dy / distance

                    # Stronger repulsion force that scales with overlap
                    overlap_x = max(
                        0, (label_i["width"] + label_j["width"]) / 2 - abs(dx)
                    )
                    overlap_y = max(
                        0, (label_i["height"] + label_j["height"]) / 2 - abs(dy)
                    )
                    overlap_severity = (overlap_x * overlap_y) / max(
                        label_i["width"] * label_i["height"], 1
                    )

                    repel_strength = (
                        k_repel * (1 + overlap_severity * 10) / max(distance, 50)
                    )

                    # Apply repulsive forces (push apart)
                    forces[i][0] -= repel_strength * unit_dx
                    forces[i][1] -= repel_strength * unit_dy
                    forces[j][0] += repel_strength * unit_dx
                    forces[j][1] += repel_strength * unit_dy

        print(f"Iteration {iteration + 1}: {overlap_count} overlaps found")

        # Debug: check if positions are actually changing
        old_positions = positions.copy()

        # Apply forces with much larger movement to match coordinate system scale
        damping = 1.0
        movement_scale = 1000  # Scale movement to match map coordinate system
        positions += forces * damping * movement_scale

        # Check how much things moved
        movement = np.sqrt(np.sum((positions - old_positions) ** 2, axis=1))
        max_movement = np.max(movement)
        avg_movement = np.mean(movement)
        print(f"  Max movement: {max_movement:.1f}, Avg movement: {avg_movement:.1f}")

        # Also check force magnitudes
        force_magnitudes = np.sqrt(np.sum(forces**2, axis=1))
        max_force = np.max(force_magnitudes)
        avg_force = np.mean(force_magnitudes)
        print(f"  Max force: {max_force:.1f}, Avg force: {avg_force:.1f}")

        # Check if we've resolved all overlaps
        if not overlaps_found:
            print(f"No overlaps detected after {iteration + 1} iterations")
            break

    print(f"Final check: {overlap_count} overlaps remaining")

    # Update label positions - no distance constraints
    for i, label in enumerate(labels):
        label["x"] = positions[i][0]
        label["y"] = positions[i][1]


def render_graph(graph_file, output_file):
    """Render the pickled graph to SVG."""
    # Load the graph
    print(f"Loading graph from {graph_file}...")
    with open(graph_file, "rb") as f:
        G = pickle.load(f)

    print(f"Loaded graph with {G.number_of_nodes()} nodes")

    # Set up the plot
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_aspect("equal")

    # Remove axes and make background transparent
    ax.set_axis_off()
    fig.patch.set_alpha(0.0)

    all_patches = []

    # Process each state polygon
    for state_code, data in G.nodes(data=True):
        wkt = data.get("wkt")
        if wkt:
            state_patches = parse_wkt_polygon(wkt)
            all_patches.extend(state_patches)

    print(f"Created {len(all_patches)} polygon patches")

    # Create patch collection
    if all_patches:
        collection = PatchCollection(
            all_patches,
            facecolor="white",
            edgecolor="#333333",  # dark grey
            linewidth=1.0,
            alpha=1.0,
        )
        ax.add_collection(collection)

        # Set plot limits based on patch extents
        ax.autoscale()

    # Get map bounds for scaling calculations
    map_bounds = ax.get_xlim(), ax.get_ylim()
    map_width = map_bounds[0][1] - map_bounds[0][0]

    # Prepare labels for force-directed positioning
    labels = []
    fontsize = 5  # Halved from 10
    for state_code, data in G.nodes(data=True):
        x = data.get("x")
        y = data.get("y")
        seats = data.get("seats", 0)
        if x is not None and y is not None and seats > 0:
            text = f"{state_code} ({seats})" if seats > 0 else state_code
            width, height = calculate_label_bbox(text, fontsize)

            # Calculate label dimensions based on map scale and actual font size
            # For monospace fonts, use more accurate character width calculation
            char_scale = map_width / 200  # Scale factor for text size vs map

            # Monospace font character width (adjusted for smaller font size)
            monospace_char_width = (
                fontsize * 0.6
            )  # Monospace chars are typically 0.6x the font size
            data_width = (
                len(text) * char_scale * (monospace_char_width / 10)
            )  # Scale relative to original 10pt
            data_height = (
                char_scale * (fontsize / 10) * 1.5
            )  # Scale height with font size

            # Adjust bboxes: 140% wider (1.6 * 1.5), 30% taller (10% + 20%)
            data_width *= 2.4
            data_height *= 1.3

            labels.append(
                {
                    "text": text,
                    "orig_x": x,
                    "orig_y": y,
                    "x": x,
                    "y": y,
                    "width": data_width,
                    "height": data_height,
                }
            )

    print(f"Applying force-directed layout to {len(labels)} labels...")

    # Debug: print some label dimensions
    for i, label in enumerate(labels[:5]):  # Print first 5
        print(
            f"Label {label['text']}: width={label['width']:.0f}, height={label['height']:.0f}"
        )

    # Apply force-directed algorithm to separate overlapping labels
    force_directed_label_layout(labels)

    # Draw labels at their adjusted positions
    for label in labels:
        # Draw bounding box as orange rectangle
        bbox_left = label["x"] - label["width"] / 2
        bbox_bottom = label["y"] - label["height"] / 2
        bbox_width = label["width"]
        bbox_height = label["height"]

        bbox_rect = patches.Rectangle(
            (bbox_left, bbox_bottom),
            bbox_width,
            bbox_height,
            linewidth=0,
            edgecolor="none",
            facecolor="orange",
            alpha=0.5,
        )
        # Note: SVG doesn't support blend modes directly through matplotlib,
        # but overlaps will be visible as darker orange areas
        ax.add_patch(bbox_rect)

        # Draw the text label with monospace font
        ax.text(
            label["x"],
            label["y"],
            label["text"],
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color="black",
            fontfamily="monospace",
        )

    # Reset plot limits to original bounds (before labels were added)
    if all_patches:
        ax.autoscale()

    # Save as SVG
    print(f"Saving to {output_file}...")
    plt.savefig(
        output_file, format="svg", transparent=True, bbox_inches="tight", pad_inches=0.1
    )
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
