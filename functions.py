import numpy as np
from scipy.spatial.distance import cdist
import random
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, value, PULP_CBC_CMD, LpStatus
import copy
import math
import matplotlib.pyplot as plt
import geopandas as gpd

def prepare_coordinates(cell_centroids, existing_coords):
    """
    Convert everything into consistent numpy arrays in projected CRS units.
    """

    cell_coords = np.vstack([np.array([geom.x, geom.y]) 
        for geom in cell_centroids.geometry])

    existing_coords = np.asarray(existing_coords)

    return cell_coords, existing_coords

# --------------------------------------------------
# STRICT FEASIBILITY CHECK (single source of truth)
# --------------------------------------------------
def is_feasible(candidate_xy, selected_idx, cell_coords, existing_coords, min_dist):

    # check against already selected
    if selected_idx:
        d_sel = np.linalg.norm(
            cell_coords[selected_idx] - candidate_xy,
            axis=1
        )
        if np.any(d_sel < min_dist):
            return False

    # check against existing gauges
    if len(existing_coords) > 0:
        d_exist = np.linalg.norm(
            existing_coords - candidate_xy,
            axis=1
        )
        if np.any(d_exist < min_dist):
            return False

    return True


# --------------------------------------------------
# GREEDY SELECTION (constraint-safe, no fallback hacks)
# --------------------------------------------------
def greedy_selection(cluster_candidates, cell_coords, existing_coords, min_dist):

    selected = {}         # cluster_id -> chosen cell index
    selected_idx = []    # list of chosen indices

    failed_clusters = {}

    for cluster_id, candidates in cluster_candidates.items():

        chosen = None

        for idx in candidates:

            candidate_xy = cell_coords[idx]

            if is_feasible(candidate_xy, selected_idx, cell_coords, existing_coords, min_dist):
                chosen = idx
                selected_idx.append(idx)
                break

        if chosen is not None:
            selected[cluster_id] = chosen
        else:
            failed_clusters[cluster_id] = "no_feasible_candidate"

    return selected, failed_clusters

# def run_greedy(cluster_candidates, cell_centroids,  existing_coords, min_dist):

#     # --- enforce consistent coordinate system ---
#     cell_coords, existing_coords = prepare_coordinates(cell_centroids, existing_coords)

#     solution = greedy_selection(cluster_candidates, cell_coords, existing_coords, min_dist)

#     return solution

# --------------------------------------------------
# SHUFFLE + GREEDY WRAPPER
# --------------------------------------------------
def shuffle_greedy(cluster_candidates, cell_coords, existing_coords, min_dist, n_iterations=500):

    best_solution = None
    best_score = (-1, float("inf"))

    cluster_keys = list(cluster_candidates.keys())

    for _ in range(n_iterations):

        # shuffle order of cluster processing
        shuffled_keys = cluster_keys.copy()
        random.shuffle(shuffled_keys)

        shuffled_candidates = {c: cluster_candidates[c] for c in shuffled_keys}

        # run greedy
        solution, failed = greedy_selection(shuffled_candidates,cell_coords,existing_coords,min_dist)

        # score MUST use full cluster set (not shuffled subset)
        s = score(solution, cluster_candidates)

        if s > best_score:
            best_score = s
            best_solution = solution

    return best_solution

def check_solution(solution,cell_coords,existing_coords,min_dist):

    selected_idx = list(solution.values())
    selected_coords = cell_coords[selected_idx]

    # --- check selected vs selected ---
    if len(selected_coords) > 1:
        d_sel = cdist(selected_coords, selected_coords)
        np.fill_diagonal(d_sel, np.inf)
        print("Min selected-selected distance:", d_sel.min())

    # --- check selected vs existing ---
    if len(existing_coords) > 0:
        d_exist = cdist(selected_coords, existing_coords)
        min_d = d_exist.min()

        print("Min selected-existing distance:", min_d)

        violations = np.where(d_exist < min_dist)

        if len(violations[0]) > 0:
            print("🚨 Violations found:")
            for i, j in zip(*violations):
                print(f"  selected {selected_idx[i]} too close to existing {j}")
                

# ── Shared scoring function ──────────────────────────────────────────────────
def evaluate_solution(selected_cells, cluster_candidates, cell_coords, existing_coords, min_dist):
    """
    Returns a dict of diagnostics for a given solution.
    """
    results = {}
    
    # 1. Coverage: how many clusters got a gauge
    results["n_covered"] = len(selected_cells)
    results["n_clusters"] = len(cluster_candidates)
    results["coverage_pct"] = 100 * len(selected_cells) / len(cluster_candidates)
    
    # 2. Candidate quality: average rank of chosen candidates
    #    (0 = top candidate chosen, higher = had to fall back)
#     ranks = []
#     for c, sel in zip(cluster_candidates.keys(), selected_cells):
#         candidates = list(cluster_candidates[c])
#         if sel in candidates:
#             ranks.append(candidates.index(sel))
#         else:
#             ranks.append(np.nan)

    ranks = []
    for c, sel in selected_cells.items():
        candidates = list(cluster_candidates[c])
        if sel in candidates:
            ranks.append(candidates.index(sel))
        else:
            ranks.append(np.nan)

    results["mean_candidate_rank"] = np.nanmean(ranks)
    results["max_candidate_rank"]  = np.nanmax(ranks)
    results["rank_details"] = dict(zip(cluster_candidates.keys(), ranks))
    
    # 3. Spacing: min distance between any two selected gauges
    selected_indices = list(selected_cells.values())
    
    
    if len(selected_indices) > 1:
        coords = cell_coords[selected_indices]
        dm = cdist(coords, coords)
        np.fill_diagonal(dm, np.inf)
        results["min_spacing_selected"] = dm.min()
        results["any_spacing_violation"] = bool((dm < min_dist).any())
    else:
        results["min_spacing_selected"] = np.nan
        results["any_spacing_violation"] = False
    
    # 4. Spacing vs existing gauges
    violations = []
    for i, idx in enumerate(selected_indices):
        pt = cell_coords[idx]

        for ex in existing_coords:
            dist = np.linalg.norm(pt - ex)
            if dist < min_dist:
                violations.append((i, dist))    
    
    results["existing_gauge_violations"] = violations
    results["any_existing_violation"] = len(violations) > 0
    
    return results


def print_evaluation(label, results, runtime, min_dist):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Coverage:          {results['n_covered']}/{results['n_clusters']} "
          f"({results['coverage_pct']:.1f}%)")
    print(f"  Mean rank chosen:  {results['mean_candidate_rank']:.2f}  "
          f"(0 = always top candidate)")
    print(f"  Max rank chosen:   {results['max_candidate_rank']:.0f}")
    print(f"  Min gauge spacing: {results['min_spacing_selected']:.0f} m  "
          f"(threshold: {min_dist} m)")
    print(f"  Spacing violation: {results['any_spacing_violation']}")
    print(f"  Existing violation:{results['any_existing_violation']}")
    print(f"  Runtime:           {runtime:.2f}s")
    if results['any_spacing_violation'] or results['any_existing_violation']:
        print("  ⚠️  INVALID SOLUTION — constraint violated")
    else:
        print("  ✅ Valid solution")
    # Show which clusters had to fall back from top candidate
    fallbacks = {c: r for c, r in results['rank_details'].items() if r > 0}
    if fallbacks:
        print(f"  Clusters using fallback candidates: {fallbacks}")



def score(solution, cluster_candidates):
    n_covered = len(solution)
    
    rank_sum = sum(list(cluster_candidates[c]).index(cell)
        for c, cell in solution.items()
        if c in cluster_candidates)
    
    return (n_covered, -rank_sum)

def get_neighbour(solution, cluster_candidates, cell_coords, existing_coords, min_dist):

    new_solution = copy.deepcopy(solution)

    # pick random cluster
    cluster_id = random.choice(list(cluster_candidates.keys()))
    candidates = cluster_candidates[cluster_id]

    current = new_solution.get(cluster_id, None)

    # try alternative candidates
    for idx in candidates:

        if idx == current:
            continue

        candidate_xy = cell_coords[idx]

        # --- check against existing ---
        if len(existing_coords) > 0:
            if np.any(np.linalg.norm(existing_coords - candidate_xy, axis=1) < min_dist):
                continue

        # --- check against current solution ---
        ok = True
        for c, sel_idx in new_solution.items():
            if c == cluster_id:
                continue
            if np.linalg.norm(cell_coords[sel_idx] - candidate_xy) < min_dist:
                ok = False
                break

        if ok:
            new_solution[cluster_id] = idx
            return new_solution

    return solution

def simulated_annealing(initial_solution, cluster_candidates, cell_coords, existing_coords, min_dist,
                        T0=1.0, Tmin=0.001, alpha=0.995, n_iter=5000):
    current = dict(initial_solution)
    current_score = score(current, cluster_candidates)
    best = dict(current)
    best_score = current_score
    T = T0

    for _ in range(n_iter):
        neighbour = get_neighbour(current, cluster_candidates, cell_coords, existing_coords, min_dist)
        neighbour_score = score(neighbour, cluster_candidates)

        # --- accept rule ---
        # coverage always dominates: only use probabilistic acceptance when coverage is equal
        if neighbour_score[0] > current_score[0]:
            # strictly more clusters covered -> always accept
            accept = True

        elif neighbour_score[0] == current_score[0]:
            # same coverage -> use rank component for SA accept rule
            delta = neighbour_score[1] - current_score[1]
            accept = delta > 0 or random.random() < math.exp(delta / T)

        else:
            # fewer clusters covered -> never accept (coverage is sacred)
            accept = False

        if accept:
            current = neighbour
            current_score = neighbour_score
            if current_score > best_score:
                best = dict(current)
                best_score = current_score

        T *= alpha
        if T < Tmin:
            break

    return best

def ilp_solution(cluster_candidates,cell_coords,existing_coords,min_dist):

    prob = LpProblem("gauge_placement", LpMinimize)

    # --------------------------------------------------
    # Decision variables
    # --------------------------------------------------
    x = {}

    for c, candidates in cluster_candidates.items():
        for rank, idx in enumerate(candidates):
            x[(c, idx)] = LpVariable(f"x_{c}_{idx}", cat="Binary")

    # --------------------------------------------------
    # Objective: minimise rank penalty
    # --------------------------------------------------
    prob += lpSum(rank * x[(c, idx)]
        for c, candidates in cluster_candidates.items()
        for rank, idx in enumerate(candidates))

    # --------------------------------------------------
    # Constraint 1: one per cluster
    # --------------------------------------------------
    for c, candidates in cluster_candidates.items():
        prob += lpSum(x[(c, idx)] for idx in candidates) == 1

    # --------------------------------------------------
    # Precompute all candidate coords
    # --------------------------------------------------
    coords = {
        idx: cell_coords[idx]
        for c in cluster_candidates
        for idx in cluster_candidates[c]}

    # --------------------------------------------------
    # Constraint 2: min distance between ALL selected points
    # --------------------------------------------------
    all_items = [(c, idx) for c, candidates in cluster_candidates.items()
                         for idx in candidates]

    for i in range(len(all_items)):
        c1, idx1 = all_items[i]

        for j in range(i + 1, len(all_items)):
            c2, idx2 = all_items[j]

            # skip same cluster pairs (already handled)
            if c1 == c2:
                continue

            if np.linalg.norm(coords[idx1] - coords[idx2]) < min_dist:
                prob += x[(c1, idx1)] + x[(c2, idx2)] <= 1

    # --------------------------------------------------
    # Constraint 3: min distance between ALL selected points
    # --------------------------------------------------                
    for c, candidates in cluster_candidates.items():
        for idx in candidates:

            candidate_xy = cell_coords[idx]

            for j, ex in enumerate(existing_coords):

                if np.linalg.norm(candidate_xy - ex) < min_dist:

                    # forbid this candidate entirely
                    prob += x[(c, idx)] == 0                


    # --------------------------------------------------
    # Solve
    # --------------------------------------------------
    prob.solve(pulp.PULP_CMD(msg=0))  # default fallback)

    print("ILP status:", LpStatus[prob.status])

    # --------------------------------------------------
    # Extract solution
    # --------------------------------------------------
    solution = {}

    for c, candidates in cluster_candidates.items():
        for idx in candidates:
            if value(x[(c, idx)]) == 1:
                solution[c] = idx

    return solution

def plot_and_check(rainfall_gdf_merged, solution,cell_centroids, catchment_Dolwen, existing_coords,min_dist,title, ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # --------------------------------------------------            
    catchment_Dolwen.boundary.plot(ax=ax, color="black")     
    rainfall_gdf_merged.plot(column="cluster",categorical=True,legend=True,ax=ax,alpha=1 , cmap='Set1')
    rainfall_gdf_merged.boundary.plot(ax=ax, color="black", linewidth=1) # Grid outlines        
        
    # --------------------------------------------------
    # Extract selected points
    # --------------------------------------------------
    selected_idx = list(solution.values())

    selected_coords = np.vstack([np.array([cell_centroids[i].x, cell_centroids[i].y])
        for i in selected_idx])
    
    existing_coords = np.asarray(existing_coords)

    # --------------------------------------------------
    # Plot base data (optional but useful)
    # --------------------------------------------------
    ax.scatter(existing_coords[:, 0], existing_coords[:, 1], c="red", marker = "X", label="Existing gauges", s=60)
    ax.scatter(selected_coords[:, 0], selected_coords[:, 1], c="black", label="Selected new gauges", s=40)

    # --------------------------------------------------
    # Draw buffer circles (true constraint radius)
    # --------------------------------------------------
    for x, y in selected_coords:
        circle = plt.Circle((x, y), min_dist, fill=False, linestyle="--", color="red", alpha=1)
        ax.add_patch(circle)

    # --------------------------------------------------
    # Final plot formatting
    # --------------------------------------------------
    ax.set_title(title)
    #ax.legend()
    ax.set_aspect("equal")

    # plt.show()
    
def trim_to_bbox_of_region_obs(obs_cube, gdf, y_coord, x_coord, buffer_km=0):

    import numpy as np
    from pyproj import Transformer

    minmax = lambda x: (np.min(x), np.max(x))

    # ---------------------------
    # Grid coordinates (BNG)
    # ---------------------------
    lats_1d = obs_cube.coord(y_coord).points
    lons_1d = obs_cube.coord(x_coord).points

    lons_2d, lats_2d = np.meshgrid(lons_1d, lats_1d)

    # ---------------------------
    # Convert to WGS84 (your current approach)
    # ---------------------------
    transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    lons_2d, lats_2d = transformer.transform(lons_2d, lats_2d)

    # ---------------------------
    # Catchment to WGS84
    # ---------------------------
    gdf = gdf.to_crs("EPSG:4326")
    bbox = gdf.total_bounds  # [minx, miny, maxx, maxy]

    # ---------------------------
    # BUFFER (NEW PART)
    # ---------------------------
    # convert km → degrees approx via projected buffer (better: do in EPSG:27700 ideally)
    buffer_deg = buffer_km / 111.0

    minx, miny, maxx, maxy = bbox
    bbox = (
        minx - buffer_deg,
        miny - buffer_deg,
        maxx + buffer_deg,
        maxy + buffer_deg
    )

    # ---------------------------
    # mask grid
    # ---------------------------
    inregion = (
        (lons_2d > bbox[0]) & (lons_2d < bbox[2]) &
        (lats_2d > bbox[1]) & (lats_2d < bbox[3])
    )

    region_inds = np.where(inregion)

    imin, imax = minmax(region_inds[0])
    jmin, jmax = minmax(region_inds[1])

    obs_cube = obs_cube[..., imin:imax+1, jmin:jmax+1]

    return obs_cube
