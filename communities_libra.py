import pickle
import sys

THRESHOLD_RANK = 1e-2
NEIGHBORS_ONLY = False
PPR_THRESHOLD_ONLY = False


def get_top_n(sub_graph, queries):
    ranks = sub_graph.personalized_pagerank(
        reset_vertices=queries,
        directed=False,
        damping=0.95,
        weights="weight",
        implementation="prpack",
    )
    ranks = sorted(
        zip([x["name"] for x in sub_graph.vs()], ranks),
        reverse=True,
        key=lambda x: x[1],
    )
    return {x[0] for x in ranks if x[1] > THRESHOLD_RANK}


if __name__ == "__main__":
    chunk_number = int(sys.argv[1])
    filename = "graph.pickle"
    with open(filename, "rb") as f:
        graph = pickle.load(f)

    filename = "nodes.pickle"
    with open(filename, "rb") as f:
        nodes = pickle.load(f)

    communities = []
    for node in nodes[chunk_number]:
        neighborhood = graph.neighborhood(node, order=1, mode="all", mindist=0)
        neighborhood = {x["name"] for x in graph.vs(neighborhood)}
        if NEIGHBORS_ONLY:
            communities.append((node, neighborhood))
        else:
            sub_g = graph.induced_subgraph(neighborhood)
            if PPR_THRESHOLD_ONLY:
                communities.append((node, get_top_n(sub_g, [node])))
            else:
                shortest_paths = sub_g.get_shortest_paths(
                    node, to=neighborhood, weights="weight", mode="all"
                )
                for path in shortest_paths:
                    communities.append((node, get_top_n(sub_g, path)))
    print(f"Done -> {chunk_number}")

    filename = f"./staging/communities-{chunk_number}.pickle"
    with open(filename, "wb") as f:
        pickle.dump(communities, f)
