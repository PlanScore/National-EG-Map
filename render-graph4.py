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
    path_parts.append(f"M {x:.2f} {y:.2f}")

    # Add L (lineto) for remaining points
    for x, y in coords[1:]:
        path_parts.append(f"L {x:.2f} {y:.2f}")

    # Close path
    path_parts.append("Z")

    # Handle holes (interior rings)
    for interior in polygon.interiors:
        interior_coords = list(interior.coords)
        if interior_coords:
            x, y = interior_coords[0]
            path_parts.append(f"M {x:.2f} {y:.2f}")
            for x, y in interior_coords[1:]:
                path_parts.append(f"L {x:.2f} {y:.2f}")
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
            "style": 'font-family:"Lucida Grande", "Lucida Sans Unicode", Arial, Helvetica, sans-serif;font-size:12px;',
        },
    )

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

                # Find largest polygon and its centroid
                largest = find_largest_polygon(geom)
                if largest:
                    cx, cy = calculate_centroid(largest)
                    state_data[state_code] = {
                        "geometry": geom,
                        "centroid": (cx, cy),
                        "label_x": data.get("x", cx),
                        "label_y": data.get("y", cy),
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

    # Calculate offsets to center the map
    offset_x = (720 - width * scale) / 2
    offset_y = (400 - height * scale) / 2

    def transform_coord(x: float, y: float) -> tuple[float, float]:
        """Transform graph coordinates to SVG viewport coordinates."""
        svg_x = (x - minx) * scale + offset_x
        # Flip Y axis (SVG has origin at top-left)
        svg_y = 400 - ((y - miny) * scale + offset_y)
        return svg_x, svg_y

    # Create a group for all state polygons
    states_group = xml.etree.ElementTree.SubElement(svg, "g", {"class": "states"})

    # Create a group for all labels (will be added after states)
    labels_group = xml.etree.ElementTree.SubElement(svg, "g", {"class": "labels"})

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
                "class": f"state state-{state_code}",
                "data-state": state_code,
                "transform": f"translate({svg_cx:.2f},{svg_cy:.2f})",
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
                    path_parts.append(f"M {rel_x:.2f} {rel_y:.2f}")
                else:
                    path_parts.append(f"L {rel_x:.2f} {rel_y:.2f}")

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
                            path_parts.append(f"M {rel_x:.2f} {rel_y:.2f}")
                        else:
                            path_parts.append(f"L {rel_x:.2f} {rel_y:.2f}")
                    path_parts.append("Z")

            path_data = " ".join(path_parts)

            # Create path element
            xml.etree.ElementTree.SubElement(
                state_group,
                "path",
                {
                    "d": path_data,
                    "fill": "#ffffff",
                    "stroke": "#333333",
                    "stroke-width": "1.0",
                },
            )

        # Add label (outside the group so it doesn't scale)
        label_svg_x, label_svg_y = transform_coord(label_x, label_y)

        text = xml.etree.ElementTree.SubElement(
            labels_group,
            "text",
            {
                "x": f"{label_svg_x:.2f}",
                "y": f"{label_svg_y:.2f}",
                "text-anchor": "middle",
                "dominant-baseline": "middle",
                "class": f"label label-{state_code}",
                "data-state": state_code,
                "style": "font-size:9px;font-weight:bold;fill:black;",
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
