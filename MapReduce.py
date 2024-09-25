import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
import json


def parse_edge(line):
    if line.startswith('%') or line.strip() == '':
        return None
    tokens = line.split()
    return int(tokens[0]), int(tokens[1])


def create_graph(edge_list):
    graph = {}
    for u, v in edge_list:
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
        graph[u].append(v)
        graph[v].append(u)
    return graph


def bfs_shortest_path(graph, start):
    queue = deque([(start, [start])])
    visited = set()
    shortest_paths = {start: [start]}

    while queue:
        node, path = queue.popleft()
        visited.add(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = path + [neighbor]
                if neighbor not in shortest_paths or len(new_path) < len(shortest_paths[neighbor]):
                    shortest_paths[neighbor] = new_path
                queue.append((neighbor, new_path))

    return shortest_paths


def map_shortest_paths(node, graph):
    return node, bfs_shortest_path(graph, node)


def plot_initial_graph(edge_list):
    G = nx.Graph()
    G.add_edges_from(edge_list)
    plt.figure(figsize=(12, 12))
    nx.draw(G, with_labels=True, node_size=700, node_color='skyblue', font_size=15)
    plt.title("Graphe des Connexions Initiales")
    plt.savefig("image/graph_initial.png")
    plt.show()


def plot_shortest_paths(graph, shortest_paths):
    G = nx.Graph()
    G.add_edges_from([(u, v) for u in graph for v in graph[u]])
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G)

    nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=15, alpha=0.3)

    for start_node, paths in shortest_paths.items():
        for end_node, path in paths.items():
            path_edges = [(u, v) for u, v in zip(path[:-1], path[1:]) if u in pos and v in pos]
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=2, edge_color='r')

    plt.title("Chemins les Plus Courts")
    plt.savefig("image/graph_short_path.png")
    plt.show()


def format_shortest_paths(kv):
    node, paths = kv
    formatted_paths = {
        "Start Node": node,
        "Paths": {
            end: {"Path": path, "Path Description": " -> ".join(map(str, path))}
            for end, path in paths.items()
        },
        "Summary": f"{len(paths)} shortest paths found for start node {node}"
    }
    return json.dumps(formatted_paths)


options = PipelineOptions()

with beam.Pipeline(options=options) as p:
    lines = p | 'ReadFile' >> beam.io.ReadFromText('./soc-dolphins.mtx')

    edges = (
        lines
        | 'ParseLines' >> beam.Map(parse_edge)
        | 'FilterNone' >> beam.Filter(lambda x: x is not None)
    )

    edge_list = edges | 'CollectEdges' >> beam.combiners.ToList()

    graph = edge_list | 'CreateGraph' >> beam.Map(create_graph)

    start_nodes = (
        edge_list
        | 'GetNodes' >> beam.FlatMap(lambda edges: set([node for edge in edges for node in edge]))
    )

    shortest_paths = (
        start_nodes
        | 'MapShortestPaths' >> beam.Map(lambda node, graph: map_shortest_paths(node, graph),
                                         graph=beam.pvalue.AsSingleton(graph))
    )

    formatted_shortest_paths = (
        shortest_paths
        | 'FormatShortestPaths' >> beam.Map(format_shortest_paths)
    )

    formatted_shortest_paths | 'WriteResults' >> beam.io.WriteToText('./result/result')

with open('./soc-dolphins.mtx', 'r') as f:
    edges = [parse_edge(line) for line in f if parse_edge(line) is not None]
    plot_initial_graph(edges)

shortest_paths = {}
with open('./result/result-00000-of-00001', 'r') as f:
    for line in f:
        data = json.loads(line.strip())
        start_node = data["Start Node"]
        paths = data["Paths"]
        shortest_paths[start_node] = {int(end_node): path_data["Path"] for end_node, path_data in paths.items()}

graph = create_graph(edges)

plot_shortest_paths(graph, shortest_paths)