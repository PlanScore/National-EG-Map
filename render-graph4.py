#!/usr/bin/env python3
"""
Render NetworkX graph with US state boundaries to SVG with transformable groups.

Takes a pickled graph file and renders state polygons in SVG format where each
state is in a transformable <g> element centered on its largest polygon's centroid.
Labels are placed at centroids but outside groups so they don't scale.

Output format matches /tmp/uschart.svg: 720x400, Lucida Grande font.
"""

from __future__ import annotations

import pickle
import sys
import xml.etree.ElementTree

import shapely.wkt
import shapely.geometry


def find_largest_polygon(geometry) -> shapely.geometry.Polygon | None:
    """Find the largest polygon by area in a geometry."""
    if geometry.geom_type == "Polygon":
        return geometry
    elif geometry.geom_type == "MultiPolygon":
        # Find the polygon with largest area
        largest = max(geometry.geoms, key=lambda p: p.area)
        return largest
    return None


def calculate_centroid(polygon: shapely.geometry.Polygon) -> tuple[float, float]:
    """Calculate the centroid of a polygon."""
    centroid = polygon.centroid
    return (centroid.x, centroid.y)


def polygon_to_svg_path(polygon: shapely.geometry.Polygon, cx: float, cy: float) -> str:
    """
    Convert a Shapely polygon to SVG path data, centered on (cx, cy).

    Args:
        polygon: The polygon to convert
        cx, cy: Center coordinates to translate the polygon to

    Returns:
        SVG path data string
    """
    # Get exterior coordinates
    coords = list(polygon.exterior.coords)
    if not coords:
        return ""

    # Start with M (moveto) for first point
    path_parts = []
    x, y = coords[0]
    path_parts.append(f"M {x:.1f} {y:.1f}")

    # Add L (lineto) for remaining points
    for x, y in coords[1:]:
        path_parts.append(f"L {x:.1f} {y:.1f}")

    # Close path
    path_parts.append("Z")

    # Handle holes (interior rings)
    for interior in polygon.interiors:
        interior_coords = list(interior.coords)
        if interior_coords:
            x, y = interior_coords[0]
            path_parts.append(f"M {x:.1f} {y:.1f}")
            for x, y in interior_coords[1:]:
                path_parts.append(f"L {x:.1f} {y:.1f}")
            path_parts.append("Z")

    return " ".join(path_parts)


def geometry_to_svg_paths(geometry, cx: float, cy: float) -> list[str]:
    """Convert a geometry (Polygon or MultiPolygon) to list of SVG path data strings."""
    paths = []

    if geometry.geom_type == "Polygon":
        path_data = polygon_to_svg_path(geometry, cx, cy)
        if path_data:
            paths.append(path_data)
    elif geometry.geom_type == "MultiPolygon":
        for poly in geometry.geoms:
            path_data = polygon_to_svg_path(poly, cx, cy)
            if path_data:
                paths.append(path_data)

    return paths


def render_graph(graph_file: str, output_file: str):
    """Render the pickled graph to SVG with transformable groups."""
    # Define offshore box positions for tiny northeastern states
    # Format: state_code -> (x, y, width, height)
    offshore_boxes = {
        "VT": (637.9, 97.1, 47.8, 29.0),
        "NH": (637.9, 127.8, 47.8, 29.0),
        "MA": (637.9, 158.5, 47.8, 29.0),
        "RI": (637.9, 189.2, 47.8, 29.0),
        "CT": (637.9, 219.9, 47.8, 29.0),
        "NJ": (637.9, 250.6, 47.8, 29.0),
        "DE": (637.9, 281.3, 47.8, 29.0),
        "MD": (637.9, 312.1, 47.8, 29.0),
    }

    # Load the graph
    print(f"Loading graph from {graph_file}...")
    with open(graph_file, "rb") as f:
        G = pickle.load(f)

    print(f"Loaded graph with {G.number_of_nodes()} nodes")

    # Create SVG root element matching /tmp/uschart.svg dimensions
    svg = xml.etree.ElementTree.Element(
        "svg",
        {
            "version": "1.1",
            "xmlns": "http://www.w3.org/2000/svg",
            "width": "720",
            "height": "400",
            "viewBox": "0 0 720 400",
        },
    )

    # Add styles
    styles = xml.etree.ElementTree.SubElement(svg, "style")
    styles.text = """
        svg {
            font-family: "Lucida Grande", "Lucida Sans Unicode", Arial, Helvetica, sans-serif;
            font-size: 12px;
        }
        .state-shape {
            stroke: #555;
            stroke-width: 1.0;
            transition: fill 0.5s ease-in-out;
        }
        .offshore-box {
            stroke: #555;
            stroke-width: 1.0;
            transition: fill 0.5s ease-in-out;
        }
        .state-label {
            font-size: 9px;
            font-weight: bold;
            text-anchor: middle;
            dominant-baseline: middle;
            transition: fill 0.5s ease-in-out;
            cursor: pointer;
            user-select: none;
            pointer-events: none;
        }
    """

    # Add description
    desc = xml.etree.ElementTree.SubElement(svg, "desc")
    desc.text = "Created with render-graph4.py"

    # Create background rectangle
    xml.etree.ElementTree.SubElement(
        svg,
        "rect",
        {
            "fill": "#ffffff",
            "x": "0",
            "y": "0",
            "width": "720",
            "height": "400",
        },
    )

    # Calculate bounds to scale graph to SVG viewport
    all_x = []
    all_y = []
    state_data = {}

    for state_code, data in G.nodes(data=True):
        wkt = data.get("wkt")
        if wkt:
            try:
                geom = shapely.wkt.loads(wkt)
                bounds = geom.bounds  # (minx, miny, maxx, maxy)
                all_x.extend([bounds[0], bounds[2]])
                all_y.extend([bounds[1], bounds[3]])

                # Apply negative buffer to geometry for centroid calculation
                buffered_geom = geom.buffer(-40000)

                # Find largest polygon and its centroid from buffered geometry
                largest = (
                    find_largest_polygon(buffered_geom)
                    if not buffered_geom.is_empty
                    else None
                )
                if largest:
                    cx, cy = calculate_centroid(largest)
                    state_data[state_code] = {
                        "geometry": geom,
                        "buffered_geometry": buffered_geom,
                        "centroid": (cx, cy),
                        "label_x": cx,
                        "label_y": cy,
                    }
                else:
                    # Fallback to original geometry if buffer is empty
                    largest = find_largest_polygon(geom)
                    if largest:
                        cx, cy = calculate_centroid(largest)
                        state_data[state_code] = {
                            "geometry": geom,
                            "buffered_geometry": None,
                            "centroid": (cx, cy),
                            "label_x": cx,
                            "label_y": cy,
                        }
            except Exception as e:
                print(f"Error processing {state_code}: {e}")

    if not all_x or not all_y:
        print("No valid geometries found")
        return

    # Calculate bounds and scaling
    minx, maxx = min(all_x), max(all_x)
    miny, maxy = min(all_y), max(all_y)
    width = maxx - minx
    height = maxy - miny

    # Calculate scale to fit in 720x400 viewport
    scale_x = 720 / width
    scale_y = 400 / height
    scale = min(scale_x, scale_y)

    # Calculate offsets to center the map (shifted 30px left)
    offset_x = (720 - width * scale) / 2 - 30
    offset_y = (400 - height * scale) / 2

    def transform_coord(x: float, y: float) -> tuple[float, float]:
        """Transform graph coordinates to SVG viewport coordinates."""
        svg_x = (x - minx) * scale + offset_x
        # Flip Y axis (SVG has origin at top-left)
        svg_y = 400 - ((y - miny) * scale + offset_y)
        return svg_x, svg_y

    # Create a group for all state polygons
    states_group = xml.etree.ElementTree.SubElement(svg, "g", {"class": "states"})

    # Create a group for offshore boxes (for tiny northeastern states)
    boxes_group = xml.etree.ElementTree.SubElement(
        svg, "g", {"class": "offshore-boxes"}
    )

    # Render offshore boxes
    for state_code, (box_x, box_y, box_width, box_height) in offshore_boxes.items():
        # Create rectangle path for the box
        box_path = (
            f"M {box_x:.1f} {box_y:.1f} "
            f"L {box_x + box_width:.1f} {box_y:.1f} "
            f"L {box_x + box_width:.1f} {box_y + box_height:.1f} "
            f"L {box_x:.1f} {box_y + box_height:.1f} Z"
        )

        xml.etree.ElementTree.SubElement(
            boxes_group,
            "path",
            {
                "d": box_path,
                "style": "fill:#eee;",
                "class": "offshore-box",
                "id": f"offshore-box-{state_code}",
            },
        )

    # Create a group for all labels (will be added after states)
    labels_group = xml.etree.ElementTree.SubElement(
        svg,
        "g",
        {"class": "labels"},
    )

    # Process each state
    for state_code, data in sorted(state_data.items()):
        geometry = data["geometry"]
        cx, cy = data["centroid"]
        label_x, label_y = data["label_x"], data["label_y"]

        # Transform centroid to SVG coordinates
        svg_cx, svg_cy = transform_coord(cx, cy)

        # Create transformable group for this state, centered on its centroid
        # The transform translates to the centroid so scaling happens from there
        state_group = xml.etree.ElementTree.SubElement(
            states_group,
            "g",
            {
                "class": "state-shape",
                "id": f"state-shape-{state_code}",
                "transform": f"translate({svg_cx:.1f},{svg_cy:.1f}) scale(1,1)",
                "style": "fill:#eee;",
            },
        )

        # Get SVG paths for this state's geometry
        # We need to translate the paths so they're centered at origin (0,0)
        # within the group (since the group itself is translated to the centroid)
        if geometry.geom_type == "Polygon":
            polygons = [geometry]
        elif geometry.geom_type == "MultiPolygon":
            polygons = list(geometry.geoms)
        else:
            continue

        for poly in polygons:
            # Get coordinates and transform them
            coords = list(poly.exterior.coords)
            if not coords:
                continue

            # Build path data, with coordinates relative to centroid
            path_parts = []
            for i, (x, y) in enumerate(coords):
                # Transform to SVG coordinates
                svg_x, svg_y = transform_coord(x, y)
                # Make relative to centroid (which is at origin in this group)
                rel_x = svg_x - svg_cx
                rel_y = svg_y - svg_cy

                if i == 0:
                    path_parts.append(f"M {rel_x:.1f} {rel_y:.1f}")
                else:
                    path_parts.append(f"L {rel_x:.1f} {rel_y:.1f}")

            path_parts.append("Z")

            # Handle holes
            for interior in poly.interiors:
                interior_coords = list(interior.coords)
                if interior_coords:
                    for i, (x, y) in enumerate(interior_coords):
                        svg_x, svg_y = transform_coord(x, y)
                        rel_x = svg_x - svg_cx
                        rel_y = svg_y - svg_cy

                        if i == 0:
                            path_parts.append(f"M {rel_x:.1f} {rel_y:.1f}")
                        else:
                            path_parts.append(f"L {rel_x:.1f} {rel_y:.1f}")
                    path_parts.append("Z")

            path_data = " ".join(path_parts)

            # Create path element
            xml.etree.ElementTree.SubElement(
                state_group,
                "path",
                {"d": path_data},
            )

        # Add label (outside the group so it doesn't scale)
        # For offshore states, place label in the offshore box
        if state_code in offshore_boxes:
            box_x, box_y, box_width, box_height = offshore_boxes[state_code]
            label_svg_x = box_x + box_width / 2
            label_svg_y = box_y + box_height / 2
        else:
            label_svg_x, label_svg_y = transform_coord(label_x, label_y)

        text = xml.etree.ElementTree.SubElement(
            labels_group,
            "text",
            {
                "x": f"{label_svg_x:.1f}",
                "y": f"{label_svg_y:.1f}",
                "class": f"state-label",
                "id": f"state-label-{state_code}",
                "style": "fill:black;",
            },
        )
        text.text = state_code

    # Write SVG to file
    print(f"Writing SVG to {output_file}...")
    tree = xml.etree.ElementTree.ElementTree(svg)
    xml.etree.ElementTree.indent(tree, space="  ")

    with open(output_file, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)

    print(f"SVG saved successfully to {output_file}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python render-graph4.py <graph.pickle> <output.svg>")
        sys.exit(1)

    graph_file = sys.argv[1]
    output_file = sys.argv[2]

    render_graph(graph_file, output_file)


if __name__ == "__main__":
    main()
