
#!/usr/bin/env python3

import pickle
import networkx as nx
from collections import deque
import pandas as pd
import random

# ============================================================
# PARAMETERS
# ============================================================

INPUT_GRAPH = "indranet_dir_graph_fix_corr_weights_s3_no_np.pkl"

OUTPUT_PATHS = "indra_drug_paths_full_with_barabasi.pkl"
OUTPUT_SUMMARY = "indra_drug_summary_full_with_barabasi.csv"
OUTPUT_RANDOM = "random_baseline.pkl"

MAX_DEPTH = 5
N_RANDOM_SAMPLES = 200

# ============================================================
# EXTENDED GENE SET
# ============================================================

EXTENDED_GENES = set([
    "ACE","ACE2","ADAM17","AGTR1","C3","C5","C5AR1","CASP1","CCL2","CGAS",
    "CHUK","CSF2","CSF3","CXCL10","CXCL8","DDX58","EGFR","EIF2AK2","FOS","IFIH1",
    "IFNA1","IFNA2","IFNAR1","IFNAR2","IFNB1","IKBKB","IKBKE","IKBKG","IL1B","IL2",
    "IL6","IL6R","IL6ST","IRAK1","IRAK4","IRF3","IRF7","IRF9","ISG15","JAK1",
    "JUN","MAP3K7","MAPK1","MAPK14","MAPK3","MAPK8","MAS1","MAVS","MX1","MX2",
    "MYD88","NFKB1","NFKBIA","NLRP3","NRP1","OAS1","OAS2","OAS3","PYCARD","RELA",
    "RIPK1","STAT1","STAT2","STAT3","STING1","SYK","TAB1","TAB2","TANK","TBK1",
    "TICAM1","TLR2","TLR3","TLR4","TLR7","TLR8","TMPRSS2","TNF","TNFRSF1A","TRAF3",
    "TRAF6","TYK2"
])

# ============================================================
# LOAD GRAPH
# ============================================================

print("Loading INDRA graph...")

with open(INPUT_GRAPH, "rb") as f:
    G = pickle.load(f)

print("Graph loaded")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# ============================================================
# DISEASE NODES
# ============================================================

seed_nodes = [n for n in EXTENDED_GENES if n in G]
print("Disease nodes found:", len(seed_nodes))

# ============================================================
# BARABÁSI: PROTEIN SPACE
# ============================================================

print("Collecting protein nodes...")

protein_nodes = [
    n for n, d in G.nodes(data=True)
    if "ns" in d and d["ns"] == "HGNC"
]

print("Total proteins:", len(protein_nodes))

# ============================================================
# BARABÁSI: DISTANCE TO DISEASE MODULE
# ============================================================

print("Precomputing distances to disease module...")

min_dist_to_S = {}

for s in seed_nodes:
    lengths = nx.single_source_shortest_path_length(G, s, cutoff=MAX_DEPTH)

    for node, d in lengths.items():
        if node not in min_dist_to_S or d < min_dist_to_S[node]:
            min_dist_to_S[node] = d

# ============================================================
# BARABÁSI: RANDOM BASELINE
# ============================================================

print("Computing random baseline...")

random_baseline = {}

for size in range(1, 21):

    samples = []

    for _ in range(N_RANDOM_SAMPLES):

        sample = random.sample(protein_nodes, size)

        d_vals = [
            min_dist_to_S.get(t, MAX_DEPTH + 1)
            for t in sample
        ]

        samples.append(sum(d_vals) / len(d_vals))

    random_baseline[size] = samples

print("Random baseline ready.")

# ============================================================
# REVERSE BFS (MECHANISTIC CORE - UNCHANGED)
# ============================================================

print("Starting reverse BFS search...")

queue = deque()
visited = {}

for gene in seed_nodes:
    queue.append((gene, [gene], 0))

paths_found = []

while queue:

    node, path, depth = queue.popleft()

    if depth >= MAX_DEPTH:
        continue

    for parent in G.predecessors(node):

        new_path = [parent] + path
        new_depth = depth + 1

        if parent not in visited or visited[parent] > new_depth:
            visited[parent] = new_depth
            queue.append((parent, new_path, new_depth))

        node_data = G.nodes[parent]

        # ====================================================
        # CHEBI FILTER
        # ====================================================

        if "ns" in node_data and node_data["ns"] == "CHEBI":

            edge_beliefs = []
            edge_corr = []
            edge_pmids = []
            edge_years = []

            edge_stmt_types = []
            edge_source_counts = []
            edge_stmt_hashes = []
            edge_english = []

            for i in range(len(new_path) - 1):

                u = new_path[i]
                v = new_path[i + 1]

                edge_data = G.get_edge_data(u, v)

                if edge_data:

                    edge_beliefs.append(edge_data.get("belief", None))
                    edge_corr.append(edge_data.get("corr_weight", None))

                    stmts = edge_data.get("statements", [])

                    stmt_types = []
                    stmt_sources = []
                    stmt_hashes = []
                    stmt_english = []

                    pmids = []
                    years = []

                    for stmt in stmts:

                        stmt_types.append(stmt.get("stmt_type", "Unknown"))
                        sc = stmt.get("source_counts", {})
                        stmt_sources.append(sc)
                        stmt_hashes.append(stmt.get("stmt_hash"))
                        stmt_english.append(stmt.get("english"))

                        ev = stmt.get("evidence", [])
                        for e in ev:
                            if isinstance(e, dict):
                                if "pmid" in e and e["pmid"]:
                                    pmids.append(e["pmid"])
                                if "year" in e and e["year"]:
                                    years.append(e["year"])

                    edge_stmt_types.append(stmt_types)
                    edge_source_counts.append(stmt_sources)
                    edge_stmt_hashes.append(stmt_hashes)
                    edge_english.append(stmt_english)

                    edge_pmids.append(pmids)
                    edge_years.append(years)

            paths_found.append({

                "molecule": parent,
                "molecule_ns": node_data.get("ns"),
                "molecule_id": node_data.get("id"),

                "gene": path[-1],

                "path_nodes": new_path,
                "path_length": new_depth,

                "beliefs": edge_beliefs,
                "corr_weights": edge_corr,

                "stmt_types": edge_stmt_types,
                "source_counts": edge_source_counts,

                "stmt_hashes": edge_stmt_hashes,
                "stmt_english": edge_english,

                "pmids": edge_pmids,
                "years": edge_years
            })

print("Paths found:", len(paths_found))

# ============================================================
# SAVE PATH DATA
# ============================================================

print("Saving path data...")

with open(OUTPUT_PATHS, "wb") as f:
    pickle.dump(paths_found, f)

# ============================================================
# SAVE RANDOM BASELINE
# ============================================================

with open(OUTPUT_RANDOM, "wb") as f:
    pickle.dump(random_baseline, f)

# ============================================================
# SUMMARY
# ============================================================

print("Creating summary...")

df = pd.DataFrame(paths_found)

summary = df.groupby("molecule").agg(
    genes_hit=("gene", "nunique"),
    paths=("gene", "count"),
    min_path_length=("path_length", "min")
).reset_index()

summary.to_csv(OUTPUT_SUMMARY, index=False)

print("Pipeline finished.")