#!/usr/bin/env python3
"""
Create NetworkX graph from remote US state boundary data.

Nodes correspond to state polygons with ISO3166_2 abbreviations as IDs.
Edges correspond to state boundary lines connecting adjacent states.
"""

import math
import pickle
import sys

import networkx
import osgeo.ogr
import osgeo.osr
import shapely.affinity
import shapely.wkt

# Enable GDAL/OGR exceptions for better error handling
osgeo.ogr.UseExceptions()
osgeo.osr.UseExceptions()


def translate_geometry(
    geom: osgeo.ogr.Geometry, dx: float, dy: float
) -> osgeo.ogr.Geometry:
    """Translate geometry by dx, dy offset using shapely."""
    wkt = geom.ExportToWkt()
    shapely_geom = shapely.wkt.loads(wkt)
    translated = shapely.affinity.translate(shapely_geom, xoff=dx, yoff=dy)
    return osgeo.ogr.CreateGeometryFromWkt(translated.wkt)


def rotate_geometry(
    geom: osgeo.ogr.Geometry, cx: float, cy: float, angle: float
) -> osgeo.ogr.Geometry:
    """Rotate geometry around point (cx, cy) by angle in radians using shapely."""
    wkt = geom.ExportToWkt()
    shapely_geom = shapely.wkt.loads(wkt)
    # Convert radians to degrees for shapely
    angle_degrees = math.degrees(angle)
    rotated = shapely.affinity.rotate(shapely_geom, angle_degrees, origin=(cx, cy))
    return osgeo.ogr.CreateGeometryFromWkt(rotated.wkt)


def scale_geometry(
    geom: osgeo.ogr.Geometry, cx: float, cy: float, scale_factor: float
) -> osgeo.ogr.Geometry:
    """Scale geometry around point (cx, cy) by scale_factor using shapely."""
    wkt = geom.ExportToWkt()
    shapely_geom = shapely.wkt.loads(wkt)
    scaled = shapely.affinity.scale(
        shapely_geom, xfact=scale_factor, yfact=scale_factor, origin=(cx, cy)
    )
    return osgeo.ogr.CreateGeometryFromWkt(scaled.wkt)


def get_largest_polygon_centroid(geometry: osgeo.ogr.Geometry) -> osgeo.ogr.Geometry:
    """Get centroid of the largest polygon in a multipolygon, or regular centroid for simple geometries."""
    if geometry.GetGeometryType() == osgeo.ogr.wkbMultiPolygon:
        largest_area = 0
        largest_polygon = None

        for i in range(geometry.GetGeometryCount()):
            polygon = geometry.GetGeometryRef(i)
            area = polygon.GetArea()
            if area > largest_area:
                largest_area = area
                largest_polygon = polygon

        if largest_polygon:
            return largest_polygon.Centroid()

    # For non-multipolygon geometries, return regular centroid
    return geometry.Centroid()


def transform_hawaii_geometry(geometry: osgeo.ogr.Geometry) -> osgeo.ogr.Geometry:
    """Transform Hawaii geometry: move 100km east and 100km south from (-710000, -1910000) then rotate 40° CCW."""
    original_centroid = geometry.Centroid()
    original_x, original_y = original_centroid.GetX(), original_centroid.GetY()

    # Move 100km east (+100000m) and 100km south (-100000m) from previous position
    target_x, target_y = -710000 + 100000, -1910000 - 100000
    hawaii_offset_x = target_x - original_x
    hawaii_offset_y = target_y - original_y

    # Create a new geometry with translated coordinates
    transformed_geometry = osgeo.ogr.CreateGeometryFromWkt(geometry.ExportToWkt())

    # Apply translation
    transformed_geometry = translate_geometry(
        transformed_geometry, hawaii_offset_x, hawaii_offset_y
    )

    # Rotate Hawaii 40 degrees counterclockwise around its centroid
    rotation_angle = math.radians(40)  # 40 degrees CCW
    translated_centroid = transformed_geometry.Centroid()
    center_x, center_y = translated_centroid.GetX(), translated_centroid.GetY()
    transformed_geometry = rotate_geometry(
        transformed_geometry, center_x, center_y, rotation_angle
    )

    return transformed_geometry


def transform_alaska_geometry(geometry: osgeo.ogr.Geometry) -> osgeo.ogr.Geometry:
    """Transform Alaska geometry: move 100km south from (-1590190, -1734924), scale to 35%, then rotate 30° CCW."""
    original_centroid = geometry.Centroid()
    original_x, original_y = original_centroid.GetX(), original_centroid.GetY()

    # Move 100km south (-100000m) from previous position
    target_x, target_y = -1590190, -1734924 - 100000
    alaska_offset_x = target_x - original_x
    alaska_offset_y = target_y - original_y

    # Create a new geometry with transformed coordinates
    transformed_geometry = osgeo.ogr.CreateGeometryFromWkt(geometry.ExportToWkt())

    # Apply translation first
    transformed_geometry = translate_geometry(
        transformed_geometry, alaska_offset_x, alaska_offset_y
    )

    # Scale Alaska to 35% around its new centroid
    translated_centroid = transformed_geometry.Centroid()
    center_x, center_y = translated_centroid.GetX(), translated_centroid.GetY()
    transformed_geometry = scale_geometry(
        transformed_geometry, center_x, center_y, 0.35
    )

    # Rotate Alaska 30 degrees counterclockwise around its centroid
    rotation_angle = math.radians(30)  # 30 degrees CCW
    scaled_centroid = transformed_geometry.Centroid()
    center_x, center_y = scaled_centroid.GetX(), scaled_centroid.GetY()
    transformed_geometry = rotate_geometry(
        transformed_geometry, center_x, center_y, rotation_angle
    )

    return transformed_geometry


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
    target_srs = osgeo.osr.SpatialReference()
    target_srs.ImportFromProj4(
        "+proj=laea +lat_0=45 +lon_0=-100 +x_0=0 +y_0=0 +a=6370997 +b=6370997 +units=m +no_defs"
    )

    # Create undirected graph
    G = networkx.Graph()

    # Track overall bounds
    min_x, min_y, max_x, max_y = (
        float("inf"),
        float("inf"),
        float("-inf"),
        float("-inf"),
    )

    # Process polygon nodes
    print("Loading polygon data...")
    polygons_ds = osgeo.ogr.Open(polygons_url)
    if not polygons_ds:
        raise RuntimeError(f"Could not open {polygons_url}")

    polygons_layer = polygons_ds.GetLayer()

    # Create coordinate transformation for polygons
    source_srs = polygons_layer.GetSpatialRef()
    coord_transform = osgeo.osr.CoordinateTransformation(source_srs, target_srs)

    node_count = 0

    # States to exclude (only PR now, since we're repositioning both HI and AK)
    excluded_states = {"PR"}

    for feature in polygons_layer:
        state_code = feature.GetField("ISO3166_2")
        if state_code and state_code not in excluded_states:
            # Get geometry and reproject
            geometry = feature.GetGeometryRef()
            if geometry:
                geometry.Transform(coord_transform)

                # Apply transformations for Hawaii and Alaska
                if state_code == "HI":
                    geometry = transform_hawaii_geometry(geometry)
                elif state_code == "AK":
                    geometry = transform_alaska_geometry(geometry)

                # Simplify geometry with 1000 map unit threshold after projection (1km)
                simplified_geometry = geometry.Simplify(1000.0)
                wkt = simplified_geometry.ExportToWkt()

                # Calculate centroid from simplified geometry (use largest polygon for multipolygons)
                centroid = get_largest_polygon_centroid(simplified_geometry)
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
    lines_ds = osgeo.ogr.Open(lines_url)
    if not lines_ds:
        raise RuntimeError(f"Could not open {lines_url}")

    lines_layer = lines_ds.GetLayer()

    # Create coordinate transformation for lines
    source_srs = lines_layer.GetSpatialRef()
    coord_transform = osgeo.osr.CoordinateTransformation(source_srs, target_srs)

    edge_count = 0

    for feature in lines_layer:
        left_state = feature.GetField("LEFT")
        right_state = feature.GetField("RIGHT")

        if (
            left_state
            and right_state
            and left_state not in excluded_states
            and right_state not in excluded_states
            and left_state in G
            and right_state in G
        ):
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
    print(f"  Connected components: {networkx.number_connected_components(G)}")
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
