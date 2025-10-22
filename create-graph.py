#!/usr/bin/env python3
"""
Create NetworkX graph from remote US state boundary data.

Nodes correspond to state polygons with ISO3166_2 abbreviations as IDs.
Edges correspond to state boundary lines connecting adjacent states.
"""

from osgeo import ogr
import networkx as nx


def main():
    # Remote data sources
    polygons_url = "/vsizip/vsicurl/https://giscollective.s3.amazonaws.com/projectlinework/times-approximate.zip/shp/Admin1_Polygons.shp"
    lines_url = "/vsizip/vsicurl/https://giscollective.s3.amazonaws.com/projectlinework/times-approximate.zip/shp/Admin1_Lines.shp"

    # Create undirected graph
    G = nx.Graph()

    # Process polygon nodes
    print("Loading polygon data...")
    polygons_ds = ogr.Open(polygons_url)
    if not polygons_ds:
        raise RuntimeError(f"Could not open {polygons_url}")

    polygons_layer = polygons_ds.GetLayer()
    node_count = 0

    for feature in polygons_layer:
        state_code = feature.GetField('ISO3166_2')
        if state_code:
            # Get geometry as WKT
            geometry = feature.GetGeometryRef()
            wkt = geometry.ExportToWkt() if geometry else None

            # Add node with WKT geometry
            G.add_node(state_code, wkt=wkt)
            node_count += 1

    print(f"Added {node_count} nodes")

    # Process line edges
    print("Loading line data...")
    lines_ds = ogr.Open(lines_url)
    if not lines_ds:
        raise RuntimeError(f"Could not open {lines_url}")

    lines_layer = lines_ds.GetLayer()
    edge_count = 0

    for feature in lines_layer:
        left_state = feature.GetField('LEFT')
        right_state = feature.GetField('RIGHT')

        if left_state and right_state and left_state in G and right_state in G:
            # Get geometry as WKT
            geometry = feature.GetGeometryRef()
            wkt = geometry.ExportToWkt() if geometry else None

            # Add edge with WKT geometry
            G.add_edge(left_state, right_state, wkt=wkt)
            edge_count += 1

    print(f"Added {edge_count} edges")

    # Print graph description
    print("\nGraph Summary:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Connected components: {nx.number_connected_components(G)}")

    # Show some example nodes
    sample_nodes = list(G.nodes())[:5]
    print(f"  Sample nodes: {sample_nodes}")

    # Show some example edges
    sample_edges = list(G.edges())[:5]
    print(f"  Sample edges: {sample_edges}")

    # Validate WKT data presence
    nodes_with_wkt = sum(1 for _, data in G.nodes(data=True) if data.get('wkt'))
    edges_with_wkt = sum(1 for _, _, data in G.edges(data=True) if data.get('wkt'))
    print(f"  Nodes with WKT: {nodes_with_wkt}/{G.number_of_nodes()}")
    print(f"  Edges with WKT: {edges_with_wkt}/{G.number_of_edges()}")

    # Show sample WKT data lengths
    if G.nodes():
        first_node = list(G.nodes(data=True))[0]
        wkt_len = len(first_node[1].get('wkt', '')) if first_node[1].get('wkt') else 0
        print(f"  Sample node WKT length: {wkt_len} chars")

    if G.edges():
        first_edge = list(G.edges(data=True))[0]
        wkt_len = len(first_edge[2].get('wkt', '')) if first_edge[2].get('wkt') else 0
        print(f"  Sample edge WKT length: {wkt_len} chars")

    return G


if __name__ == "__main__":
    graph = main()