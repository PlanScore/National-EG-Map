#!/usr/bin/env python3
"""
Create NetworkX graph from remote US state boundary data.

Nodes correspond to state polygons with ISO3166_2 abbreviations as IDs.
Edges correspond to state boundary lines connecting adjacent states.
"""

import sys
import pickle
from osgeo import ogr, osr
import networkx as nx

# Enable GDAL/OGR exceptions for better error handling
ogr.UseExceptions()
osr.UseExceptions()


def create_graph():
    # Remote data sources
    polygons_url = "/vsizip/vsicurl/https://giscollective.s3.amazonaws.com/projectlinework/times-approximate.zip/shp/Admin1_Polygons.shp"
    lines_url = "/vsizip/vsicurl/https://giscollective.s3.amazonaws.com/projectlinework/times-approximate.zip/shp/Admin1_Lines.shp"

    # Congressional seats from 2020 apportionment
    congressional_seats = {
        "AL": 7,
        "AK": 1,
        "AZ": 9,
        "AR": 4,
        "CA": 52,
        "CO": 8,
        "CT": 5,
        "DE": 1,
        "FL": 28,
        "GA": 14,
        "HI": 2,
        "ID": 2,
        "IL": 17,
        "IN": 9,
        "IA": 4,
        "KS": 4,
        "KY": 6,
        "LA": 6,
        "ME": 2,
        "MD": 8,
        "MA": 9,
        "MI": 13,
        "MN": 8,
        "MS": 4,
        "MO": 8,
        "MT": 2,
        "NE": 3,
        "NV": 4,
        "NH": 2,
        "NJ": 12,
        "NM": 3,
        "NY": 26,
        "NC": 14,
        "ND": 1,
        "OH": 15,
        "OK": 5,
        "OR": 6,
        "PA": 17,
        "RI": 2,
        "SC": 7,
        "SD": 1,
        "TN": 9,
        "TX": 38,
        "UT": 4,
        "VT": 1,
        "VA": 11,
        "WA": 10,
        "WV": 2,
        "WI": 8,
        "WY": 1,
    }

    # Create coordinate transformation to ESRI:102004 (Lambert Azimuthal Equal Area)
    target_srs = osr.SpatialReference()
    target_srs.ImportFromProj4(
        "+proj=laea +lat_0=45 +lon_0=-100 +x_0=0 +y_0=0 +a=6370997 +b=6370997 +units=m +no_defs"
    )

    # Create undirected graph
    G = nx.Graph()

    # Track overall bounds
    min_x, min_y, max_x, max_y = (
        float("inf"),
        float("inf"),
        float("-inf"),
        float("-inf"),
    )

    # Process polygon nodes
    print("Loading polygon data...")
    polygons_ds = ogr.Open(polygons_url)
    if not polygons_ds:
        raise RuntimeError(f"Could not open {polygons_url}")

    polygons_layer = polygons_ds.GetLayer()

    # Create coordinate transformation for polygons
    source_srs = polygons_layer.GetSpatialRef()
    coord_transform = osr.CoordinateTransformation(source_srs, target_srs)

    node_count = 0

    for feature in polygons_layer:
        state_code = feature.GetField("ISO3166_2")
        if state_code:
            # Get geometry and reproject
            geometry = feature.GetGeometryRef()
            if geometry:
                geometry.Transform(coord_transform)
                # Simplify geometry with 1000 map unit threshold after projection (1km)
                simplified_geometry = geometry.Simplify(1000.0)
                wkt = simplified_geometry.ExportToWkt()

                # Calculate centroid from simplified geometry
                centroid = simplified_geometry.Centroid()
                center_x = centroid.GetX()
                center_y = centroid.GetY()

                # Update bounds using simplified geometry
                envelope = simplified_geometry.GetEnvelope()
                min_x = min(min_x, envelope[0])
                max_x = max(max_x, envelope[1])
                min_y = min(min_y, envelope[2])
                max_y = max(max_y, envelope[3])
            else:
                wkt = None
                center_x = None
                center_y = None

            # Get congressional seats for this state
            seats = congressional_seats.get(state_code, 0)

            # Add node with reprojected WKT geometry, center coordinates, and seats
            G.add_node(state_code, wkt=wkt, x=center_x, y=center_y, seats=seats)
            node_count += 1

    print(f"Added {node_count} nodes")

    # Process line edges
    print("Loading line data...")
    lines_ds = ogr.Open(lines_url)
    if not lines_ds:
        raise RuntimeError(f"Could not open {lines_url}")

    lines_layer = lines_ds.GetLayer()

    # Create coordinate transformation for lines
    source_srs = lines_layer.GetSpatialRef()
    coord_transform = osr.CoordinateTransformation(source_srs, target_srs)

    edge_count = 0

    for feature in lines_layer:
        left_state = feature.GetField("LEFT")
        right_state = feature.GetField("RIGHT")

        if left_state and right_state and left_state in G and right_state in G:
            # Get geometry and reproject
            geometry = feature.GetGeometryRef()
            if geometry:
                geometry.Transform(coord_transform)
                # Simplify geometry with 1000 map unit threshold after projection (1km)
                simplified_geometry = geometry.Simplify(1000.0)
                wkt = simplified_geometry.ExportToWkt()

                # Update bounds using simplified geometry
                envelope = simplified_geometry.GetEnvelope()
                min_x = min(min_x, envelope[0])
                max_x = max(max_x, envelope[1])
                min_y = min(min_y, envelope[2])
                max_y = max(max_y, envelope[3])
            else:
                wkt = None

            # Add edge with reprojected WKT geometry
            G.add_edge(left_state, right_state, wkt=wkt)
            edge_count += 1

    print(f"Added {edge_count} edges")

    # Print graph description
    print("\nGraph Summary:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Connected components: {nx.number_connected_components(G)}")
    print("  Coordinate system: ESRI:102004 (Lambert Azimuthal Equal Area)")
    if min_x != float("inf"):
        print(
            f"  Data bounds: ({min_x:.0f}, {min_y:.0f}) to ({max_x:.0f}, {max_y:.0f})"
        )
        width = max_x - min_x
        height = max_y - min_y
        print(f"  Data extent: {width:.0f} x {height:.0f} units")

    # Show some example nodes
    sample_nodes = list(G.nodes())[:5]
    print(f"  Sample nodes: {sample_nodes}")

    # Show some example edges
    sample_edges = list(G.edges())[:5]
    print(f"  Sample edges: {sample_edges}")

    # Validate WKT data presence
    nodes_with_wkt = sum(1 for _, data in G.nodes(data=True) if data.get("wkt"))
    edges_with_wkt = sum(1 for _, _, data in G.edges(data=True) if data.get("wkt"))
    print(f"  Nodes with WKT: {nodes_with_wkt}/{G.number_of_nodes()}")
    print(f"  Edges with WKT: {edges_with_wkt}/{G.number_of_edges()}")

    # Show sample WKT data lengths and center coordinates
    if G.nodes():
        first_node = list(G.nodes(data=True))[0]
        wkt_len = len(first_node[1].get("wkt", "")) if first_node[1].get("wkt") else 0
        center_x = first_node[1].get("x")
        center_y = first_node[1].get("y")
        print(f"  Sample node WKT length: {wkt_len} chars")
        if center_x is not None and center_y is not None:
            print(f"  Sample node center: ({center_x:.0f}, {center_y:.0f})")

    if G.edges():
        first_edge = list(G.edges(data=True))[0]
        wkt_len = len(first_edge[2].get("wkt", "")) if first_edge[2].get("wkt") else 0
        print(f"  Sample edge WKT length: {wkt_len} chars")

    # Validate center coordinates and congressional seats
    nodes_with_centers = sum(
        1
        for _, data in G.nodes(data=True)
        if data.get("x") is not None and data.get("y") is not None
    )
    nodes_with_seats = sum(
        1
        for _, data in G.nodes(data=True)
        if data.get("seats") is not None and data.get("seats") > 0
    )
    total_seats = sum(data.get("seats", 0) for _, data in G.nodes(data=True))
    print(
        f"  Nodes with center coordinates: {nodes_with_centers}/{G.number_of_nodes()}"
    )
    print(f"  Nodes with congressional seats: {nodes_with_seats}/{G.number_of_nodes()}")
    print(f"  Total congressional seats: {total_seats}")

    return G


def main():
    if len(sys.argv) != 2:
        print("Usage: python create-graph.py <output.pickle>")
        sys.exit(1)

    output_file = sys.argv[1]
    graph = create_graph()

    # Save graph to pickle file
    print(f"\nSaving graph to {output_file}...")
    with open(output_file, "wb") as f:
        pickle.dump(graph, f)

    print(f"Graph saved successfully to {output_file}")
    return graph


if __name__ == "__main__":
    main()
