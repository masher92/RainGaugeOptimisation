import numpy as np
from scipy.spatial.distance import cdist
import random
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, value, PULP_CBC_CMD, LpStatus
import copy
import math
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from scipy.spatial import cKDTree
from pyproj import Transformer
from affine import Affine
from rasterio.features import geometry_mask
import rasterio
from rasterio.mask import mask

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

def find_backup_30m_locations(selected_df, zone_raster, combined_mask, labels_30m_full, dem_arr, slope_arr, valid_mask_30m,
                              existing_coords, transform, diagnostics, min_dist, zone_id, n_new, plot=False):
    
    # print(zone_id)
    n_alternatives = 5

    cell = selected_df[selected_df["zone_id"] == zone_id].iloc[0]
    rr = cell["row"]
    cc = cell["col"]
    new_gauge_num = cell["new_gauge_num"]

    mask = (zone_raster == zone_id) & combined_mask
    rows, cols = np.where(mask)

    rmin, rmax = rows.min(), rows.max()
    cmin, cmax = cols.min(), cols.max()

    rr_local = rr - rmin
    cc_local = cc - cmin

    subset_clusters = labels_30m_full[rmin:rmax+1, cmin:cmax+1]
    subset_dem = dem_arr[rmin:rmax+1, cmin:cmax+1]
    subset_slope = slope_arr[rmin:rmax+1, cmin:cmax+1]
    subset_mask = mask[rmin:rmax+1, cmin:cmax+1]

    clusters_plot = np.ma.masked_where(~subset_mask, subset_clusters)
    dem_plot = np.ma.masked_where(~subset_mask, subset_dem)
    slope_plot = np.ma.masked_where(~subset_mask, subset_slope)

    dem_vmin = np.nanmin(dem_arr[valid_mask_30m])
    dem_vmax = np.nanmax(dem_arr[valid_mask_30m])
    slope_vmin = np.nanmin(slope_arr[valid_mask_30m])
    slope_vmax = np.nanmax(slope_arr[valid_mask_30m])

    # --- Distance to nearest gauge ---
    other_gauge_coords = np.array(
        existing_coords.tolist() if hasattr(existing_coords, "tolist") else list(existing_coords))
    other_new_gauges = selected_df[selected_df["zone_id"] != zone_id][["x", "y"]].values

    if len(other_new_gauges):
        all_gauge_coords = np.vstack([other_gauge_coords, other_new_gauges])
    else:
        all_gauge_coords = other_gauge_coords

    local_rows, local_cols = np.where(subset_mask)
    global_rows = local_rows + rmin
    global_cols = local_cols + cmin

    xs, ys = rasterio.transform.xy(transform, global_rows, global_cols)
    xs, ys = np.array(xs), np.array(ys)

    pixel_coords = np.column_stack([xs, ys])

    dists = np.sqrt(
        ((pixel_coords[:, None, 0] - all_gauge_coords[None, :, 0]) ** 2) +
        ((pixel_coords[:, None, 1] - all_gauge_coords[None, :, 1]) ** 2))

    min_dist_to_gauge = dists.min(axis=1)

    dist_full = np.full(subset_mask.shape, np.nan)
    dist_full[local_rows, local_cols] = min_dist_to_gauge
    dist_plot = np.ma.masked_where(~subset_mask, dist_full)

    # --- Ranked candidates function ---
    def get_ranked_feasible_alternatives(zone_id, diagnostics, transform, all_gauge_coords,
                                         min_dist, n_alternatives=5):

        ranked = diagnostics[zone_id]["candidates_ranked"].copy()

        xs, ys = rasterio.transform.xy(transform, ranked["row"].values, ranked["col"].values)
        ranked["x"] = xs
        ranked["y"] = ys

        if len(all_gauge_coords):
            dist_matrix = np.sqrt(
                ((ranked["x"].values[:, None] - all_gauge_coords[None, :, 0]) ** 2) +
                ((ranked["y"].values[:, None] - all_gauge_coords[None, :, 1]) ** 2)
            )
            ranked["dist_to_nearest_gauge"] = dist_matrix.min(axis=1)
        else:
            ranked["dist_to_nearest_gauge"] = np.inf

        ranked["feasible"] = ranked["dist_to_nearest_gauge"] >= min_dist

        result = ranked[ranked["feasible"]].head(n_alternatives).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)
        return result

    ranked = get_ranked_feasible_alternatives(
        zone_id, diagnostics, transform, all_gauge_coords, min_dist, n_alternatives)
    ranked['new_gauge_num'] = new_gauge_num

    ranked["row_local"] = ranked["row"] - rmin
    ranked["col_local"] = ranked["col"] - cmin

    # --- Figure with 5 panels ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes=axes.flatten()

    panels = [
        (axes[0], clusters_plot, "tab10", None, None, f"30m clusters — cell {zone_id}"),
        (axes[1], dem_plot, "terrain", dem_vmin, dem_vmax, f"Elevation (m) — cell {zone_id}"),
        (axes[2], slope_plot, "viridis", slope_vmin, slope_vmax, f"Slope (°) — cell {zone_id}"),
        (axes[3], dist_plot, "RdYlGn", None, None, f"Distance to nearest gauge (m)"),]

    # --- Local panels ---
    for ax, data, cmap, vmin, vmax, title in panels:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for _, r in ranked.iterrows():
            if r["rank"] == 1:
                ax.scatter(r["col_local"], r["row_local"],
                           c="red", edgecolors="black",
                           s=180, marker="*", zorder=6)
            else:
                ax.text(
                    r["col_local"], r["row_local"], str(int(r["rank"])),
                    ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="circle,pad=0.25",
                              fc="white", ec="black", lw=0.8),
                    zorder=5,
                )

    # --- Distance contour ---
    cs = axes[3].contour(dist_plot, levels=[min_dist], colors="black", linewidths=1.5)
    axes[3].clabel(cs, fmt=f"{min_dist:.0f} m")

    # ==========================================================
    # 🌍 FULL CATCHMENT PANEL
    # ==========================================================
    ax_full = axes[4]

    # Compute extent from transform
    nrows, ncols = dem_arr.shape
    x0, y0 = rasterio.transform.xy(transform, 0, 0)
    x1, y1 = rasterio.transform.xy(transform, nrows-1, ncols-1)

    extent = [x0, x1, y1, y0]

    im_full = ax_full.imshow( np.ma.masked_where(~valid_mask_30m, dem_arr), cmap="terrain", vmin=dem_vmin, vmax=dem_vmax,
                             extent=extent)

    ax_full.set_title("Gauge location (full catchment)")
    ax_full.axis("off")

    # Highlight current zone
    zone_outline = np.zeros_like(zone_raster, dtype=float)
    zone_outline[zone_raster == zone_id] = 1

    # ax_full.contour(zone_outline, levels=[0.5], colors="red", linewidths=1.5, extent=extent)

    # Existing gauges
    if len(existing_coords):
        ax_full.scatter(existing_coords[:, 0], existing_coords[:, 1],
                        c="black", s=20, label="Existing", zorder=3)

    # All new gauges
    all_selected = selected_df[["x", "y"]].values
    if len(all_selected):
        ax_full.scatter(all_selected[:, 0], all_selected[:, 1],
                        c="blue", s=30, label="New", zorder=4)

    # Current zone candidates
    for _, r in ranked.iterrows():
        if r["rank"] == 1:
            ax_full.scatter(r["x"], r["y"],
                            c="red", edgecolors="black",
                            s=120, marker="*", zorder=6)
        else:
            ax_full.text( r["x"], r["y"], str(int(r["rank"])), ha="center", va="center", fontsize=8, fontweight="bold", 
                         bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="black"), zorder=5)

    ax_full.legend(loc="lower left")
    fig.colorbar(im_full, ax=ax_full, fraction=0.046, pad=0.04)

    n_used = 5
    for i in range(n_used, len(axes)):
        axes[i].axis('off')
    plt.tight_layout()

    fig.savefig(f'Outputs/{n_new}Gauges/Gauge_{new_gauge_num}_Alternatives.png')

    if plot == True:
        plt.show()
    else:
        plt.close(fig)   # prevents inline auto-display when plot=False

    return ranked



def get_grid_data(obj, y_coord='projection_y_coordinate', x_coord='projection_x_coordinate'):
    """Return (data2d, x1d, y1d) from an iris Cube or xarray DataArray."""
    if hasattr(obj, 'coord'):  # iris cube
        y1d = obj.coord(y_coord).points
        x1d = obj.coord(x_coord).points
        data = obj.data
        data2d = data.filled(np.nan) if np.ma.is_masked(data) else np.asarray(data)
    else:  # xarray DataArray
        # handle either rioxarray's default 'x'/'y' or named projection coords
        y_name = y_coord if y_coord in obj.coords else 'y'
        x_name = x_coord if x_coord in obj.coords else 'x'
        y1d = obj[y_name].values
        x1d = obj[x_name].values
        data2d = obj.values
    return np.asarray(data2d, dtype=float), x1d, y1d


def sample_at_points(data2d, x1d, y1d, gdf, crs="EPSG:27700"):
    """Nearest-neighbour sample of a 2D grid at point locations, skipping NaN cells
    so a gauge near the catchment edge doesn't snap to a masked-out cell."""
    gdf = gdf.to_crs(crs)
    xx, yy = np.meshgrid(x1d, y1d)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    flat_data = data2d.ravel()
    valid = ~np.isnan(flat_data)
    tree = cKDTree(grid_points[valid])
    values = np.array([flat_data[valid][tree.query([geom.x, geom.y])[1]]
                        for geom in gdf.geometry])
    return values


def plot_spatial(ax, data2d, x1d, y1d, catchment_gdf, gauges_existing=None, gauges_new=None,
                 title='', units='', cmap='Blues'):
    
    im = ax.pcolormesh(x1d, y1d, data2d, cmap=cmap, shading='auto')
    
    catchment_gdf.to_crs("EPSG:27700").boundary.plot(ax=ax, color='black', linewidth=1.2)

    if gauges_existing is not None:
        g = gauges_existing.to_crs("EPSG:27700")
        ax.scatter(g.geometry.x, g.geometry.y, marker='o', s=70,
                   facecolor='white', edgecolor='black',
                   linewidth=1.2, label='Existing gauges', zorder=5)

    if gauges_new is not None:
        g = gauges_new.to_crs("EPSG:27700")
        ax.scatter(g.geometry.x, g.geometry.y, marker='*', s=150,
                   facecolor='red', edgecolor='black',
                   linewidth=0.8, label='Proposed gauges', zorder=5)

    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)

    # ✅ FIX HERE
    ax.figure.colorbar(im, ax=ax, shrink=0.8, label=units)

    ax.axis("off")


def plot_histogram(ax, data2d, existing_values=None, new_values=None, title='', xlabel=''):
    flat = data2d[~np.isnan(data2d)]
    ax.hist(flat, bins=30, color='steelblue', edgecolor='white', alpha=0.8, label='Catchment cells')

    if existing_values is not None:
        for i, v in enumerate(existing_values):
            ax.axvline(v, color='black', linestyle='--', linewidth=1.5,
                       label='Existing gauges' if i == 0 else None)
    if new_values is not None:
        for i, v in enumerate(new_values):
            ax.axvline(v, color='red', linestyle=':', linewidth=1.5,
                       label='Proposed gauges' if i == 0 else None)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Number of grid cells')
    ax.set_title(title)
    ax.legend(fontsize=8)

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
    bbox = (minx - buffer_deg, miny - buffer_deg, maxx + buffer_deg, maxy + buffer_deg)

    # ---------------------------
    # mask grid
    # ---------------------------
    inregion = ((lons_2d > bbox[0]) & (lons_2d < bbox[2]) & (lats_2d > bbox[1]) & (lats_2d < bbox[3]))

    region_inds = np.where(inregion)

    imin, imax = minmax(region_inds[0])
    jmin, jmax = minmax(region_inds[1])

    obs_cube = obs_cube[..., imin:imax+1, jmin:jmax+1]

    return obs_cube   

def mask_to_catchment(obs_cube, gdf, y_coord='projection_y_coordinate', x_coord='projection_x_coordinate'):
    """
    Mask cells outside the catchment polygon boundary (not just its bbox).
    Assumes obs_cube coords are in the same CRS as EPSG:27700 (BNG),
    which is standard for CEH-GEAR.
    """
    # Reproject catchment to the grid's native CRS (BNG) - cheap, few vertices
    gdf_bng = gdf.to_crs("EPSG:27700")
    geoms = gdf_bng.geometry.values

    y = obs_cube.coord(y_coord).points
    x = obs_cube.coord(x_coord).points

    dx = x[1] - x[0]
    dy = y[1] - y[0]  # note: often negative if y runs north->south

    # Affine transform mapping (col, row) -> (x, y), pixel corners at cell edges
    transform = Affine.translation(x[0] - dx / 2, y[0] - dy / 2) * Affine.scale(dx, dy)

    ny, nx = len(y), len(x)

    # geometry_mask default: True = outside the shapes (i.e. cells to mask out)
    outside_mask = geometry_mask(geoms, out_shape=(ny, nx), transform=transform, invert=False)

    # Broadcast the 2D mask across any leading (e.g. time) dimensions
    full_mask = np.broadcast_to(outside_mask, obs_cube.shape)

    masked_data = np.ma.masked_array(obs_cube.data, mask=full_mask.copy())  # .copy() to make writeable
    obs_cube.data = masked_data
    return obs_cube

def sample_landcover_at_gauges(clipped, transform, gdf, raster_crs, class_col_name='landcover_class'):
    """
    Sample a clipped landcover raster at gauge point locations.
    gdf: GeoDataFrame of gauge points (any CRS - will be reprojected to match raster)
    clipped: 2D array from rasterio.mask.mask (already band-indexed, e.g. clipped[0])
    transform: the affine transform returned alongside `clipped`
    """
    gdf = gdf.to_crs(raster_crs)
    coords = [(geom.x, geom.y) for geom in gdf.geometry]

    # rasterio doesn't have a direct "sample an array" method - convert
    # coords to row/col via the transform ourselves (fast, exact)
    rows, cols = rasterio.transform.rowcol(transform, [c[0] for c in coords], [c[1] for c in coords])

    values = []
    for r, c in zip(rows, cols):
        if 0 <= r < clipped.shape[0] and 0 <= c < clipped.shape[1]:
            values.append(clipped[r, c])
        else:
            values.append(np.nan)  # gauge falls outside the clipped raster extent

    gdf = gdf.copy()
    gdf[class_col_name] = values
    return gdf