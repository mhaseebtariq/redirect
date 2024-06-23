import pickle
import sys

THRESHOLD_RANK = 1e-8
THRESHOLD_SIZE = 150


def get_top_n(sub_g, queries):
    ranks = sub_g.personalized_pagerank(
        reset_vertices=queries,
        directed=False,
        damping=0.85,
        weights="weight",
        implementation="prpack",
    )
    ranks = sorted(
        zip(
            [x["name"] for x in sub_g.vs()], ranks
        ), reverse=True, key=lambda x: x[1]
    )[:THRESHOLD_SIZE]
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
    for index, node in enumerate([str(x) for x in nodes[chunk_number]]):
        neighborhood = graph.neighborhood(node, order=2, mode="all", mindist=0)
        communities.append(
            (node, get_top_n(graph.induced_subgraph(neighborhood), [node]))
        )
    print(f"Done -> {chunk_number}")

    filename = f"./staging/communities-{chunk_number}.pickle"
    with open(filename, "wb") as f:
        pickle.dump(communities, f)
