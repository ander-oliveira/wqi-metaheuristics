"""Standalone IQC map generation mode (no metaheuristic execution).

Loads existing walkability datasets (mode-1 outputs), rebuilds the map context
(street network, green and water areas — downloaded from OSM on first use and
cached under ``data/cache``), and generates the IQC heatmaps exactly like the
original pipeline does.

Optionally overlays the installed POIs of a metaheuristic result
(``results/<method>/.../summary/global_best_allocation.csv``) as black dots and
also generates a second heatmap with the IQC recomputed after applying that
allocation (one objective evaluation — no metaheuristic is run).
"""

from .common import *

from metaheuristics.core.budget import POI_DIMENSION_COLUMNS
from metaheuristics.core.evaluation import (
    allocation_items_to_candidate_matrix,
    build_final_indicator_matrix_nd,
    build_objective_state_nd,
    get_available_dimensions,
    objective_function,
)

from .data_sources import get_green_and_water_areas, get_pois
from .features import map_poi_colors_and_types
from .network_ops import get_center_node
from .utils import build_analysis_base_dir
from .visualization import plot_walkability_heatmap


METHOD_DIRS = ['ils', 'grasp', 'brkga', 'pso', 'hybrid_grasp_vns_pr']
METHOD_LABELS = {
    'ils': 'ILS',
    'grasp': 'GRASP',
    'brkga': 'BRKGA',
    'pso': 'PSO',
    'hybrid_grasp_vns_pr': 'Hybrid GRASP + VNS + PR',
}
POI_COLORS = {
    'bar': '#FDBF6F', 'cafe': '#A6761D', 'fast_food': '#FF7F00',
    'restaurant': '#D22426', 'college': '#377EB8', 'kindergarten': '#984EA3',
    'library': '#4DAF4A', 'school': '#377EB8', 'university': '#377EB8',
    'bicycle_parking': '#CCCCCC', 'bank': '#FFFF33', 'clinic': '#FB8072',
    'dentist': '#BEBADA', 'doctors': '#FB8072', 'hospital': '#E31A1C',
    'pharmacy': '#8DD3C7', 'cinema': '#BC80BD', 'theatre': '#BC80BD',
    'hotel': '#B3DE69',
    'gym': '#440154', 'fitness_centre': '#440154', 'fitness_center': '#440154',
    'supermarket': '#31688E', 'convenience': '#35B779',
    'bakery': '#FDE724', 'greengrocer': '#6DCD59', 'grocery': '#35B779'
}


def _get_or_build_graph(place: tuple, distance: float, network_type: str = 'walk'):
    """Projected walk graph for the area, cached on disk after the first download."""
    cache_dir = 'data/cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = f"graph_{place[0]:.6f}_{place[1]:.6f}_{distance}_{network_type}"
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")

    if os.path.exists(cache_file):
        print("Loading street network graph from cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print("Downloading walkable street network graph from OSM...")
    graph = ox.graph_from_point(place, dist=distance, network_type=network_type)
    graph = ox.projection.project_graph(graph)

    with open(cache_file, 'wb') as f:
        pickle.dump(graph, f)
    return graph


def _get_colored_pois(place: tuple, distance: float, target_crs) -> gpd.GeoDataFrame:
    """Existing POIs for map overlays, colored with the same palette as the main pipeline."""
    gdf_pois = get_pois(place, distance, target_crs)
    return map_poi_colors_and_types(gdf_pois, POI_COLORS)


def _find_datasets(key_location: str, profile_keys: list,
                   h3_resolution: int, distance: int) -> list:
    """Existing (profile, walkability CSV, hex-time matrix CSV) triples."""
    found = []
    suffix_template = f"_res_{h3_resolution}_dist{distance}.csv"
    for profile_key in profile_keys:
        base_dir = build_analysis_base_dir(key_location, profile_key, h3_resolution, distance)
        walkability_dir = os.path.join(base_dir, 'csv', 'walkability_index')
        if not os.path.isdir(walkability_dir):
            continue
        expected_suffix = f"_walkability_index_{profile_key}{suffix_template}"
        for file_name in sorted(os.listdir(walkability_dir)):
            if not file_name.endswith(expected_suffix):
                continue
            file_path = os.path.join(walkability_dir, file_name)
            matrix_path = os.path.join(
                walkability_dir, file_name.replace('_walkability_index_', '_hex_time_matrix_'))
            found.append({
                'profile_key': profile_key,
                'base_dir': base_dir,
                'file_path': file_path,
                'hex_time_matrix_path': matrix_path if os.path.isfile(matrix_path) else None,
            })
            break  # one dataset per profile
    return found


def _find_latest_allocation(method: str, key_location: str, profile_key: str):
    """Newest global_best_allocation.csv for (method, location, profile)."""
    root = os.path.join('results', method, key_location, profile_key)
    if not os.path.isdir(root):
        return None
    candidates = []
    for dataset_dir in sorted(os.listdir(root)):
        runs_root = os.path.join(root, dataset_dir)
        if not os.path.isdir(runs_root):
            continue
        for run_dir in sorted(os.listdir(runs_root)):
            alloc = os.path.join(runs_root, run_dir, 'summary', 'global_best_allocation.csv')
            if os.path.isfile(alloc):
                candidates.append((run_dir, alloc))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])  # run dirs are timestamped
    return candidates[-1][1]


def _ask_overlay_method() -> Optional[str]:
    """Which method's best allocation to overlay as installed POIs (or none)."""
    available = []
    for method in METHOD_DIRS:
        if os.path.isdir(os.path.join('results', method)):
            available.append(method)

    print("\nOverlay installed POIs from a metaheuristic result?")
    print("0 - No overlay (baseline IQC maps only)")
    for idx, method in enumerate(available, start=1):
        print(f"{idx} - {method} (latest run per location/profile)")

    while True:
        choice = input(f"Enter option [0-{len(available)}]: ").strip()
        if choice == '0' or choice == '':
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(available):
            return available[int(choice) - 1]
        print("Invalid option.")


def _installed_pois_gdf(df_allocation: pd.DataFrame, target_crs) -> gpd.GeoDataFrame:
    """Installed POIs as points at hexagon centers, with total quantity per hexagon."""
    grouped = (
        df_allocation.groupby('h3_id', as_index=False)['quantity'].sum()
    )
    latlng = [h3.cell_to_latlng(h3_id) for h3_id in grouped['h3_id']]
    gdf = gpd.GeoDataFrame(
        grouped,
        geometry=[Point(lng, lat) for lat, lng in latlng],
        crs='EPSG:4326',
    )
    return gdf.to_crs(target_crs)


def _recompute_iqc_with_allocation(df_walkability: pd.DataFrame,
                                   df_hex_time_matrix: pd.DataFrame,
                                   df_allocation: pd.DataFrame) -> pd.DataFrame:
    """Apply the allocation and recompute IQC for every hexagon (one evaluation)."""
    dimensions = get_available_dimensions(df_walkability, POI_DIMENSION_COLUMNS)
    state = build_objective_state_nd(
        df_walkability=df_walkability,
        df_hex_time_matrix=df_hex_time_matrix,
        candidate_dimensions=dimensions,
    )
    candidate_matrix = allocation_items_to_candidate_matrix(
        allocation_items=df_allocation.to_dict('records'),
        objective_state=state,
    )
    final_matrix = build_final_indicator_matrix_nd(
        candidate_matrix=candidate_matrix,
        objective_state=state,
    )
    result = objective_function(final_indicator_matrix=final_matrix)
    print(f"Recomputed IQC after allocation: sum={result['objective_value']:.4f}")
    return pd.DataFrame({'h3_id': state.h3_ids, 'IQC': result['iqc_values']})


def run_iqc_map_generation(selected_locations: list,
                           profile_keys: list,
                           h3_resolution: int,
                           distance: int,
                           network_type: str = 'walk') -> None:
    """Entry point for execution mode 4: IQC maps only, no metaheuristics."""
    overlay_method = _ask_overlay_method()

    for location_idx, (central_point, location, _dem_path, key_location) in enumerate(selected_locations, start=1):
        print(f"\n=== Location [{location_idx}/{len(selected_locations)}]: {location} ===")

        datasets = _find_datasets(key_location, profile_keys, h3_resolution, distance)
        if not datasets:
            print("No existing walkability datasets found for this location. Skipping.")
            continue

        graph = _get_or_build_graph(central_point, distance, network_type)
        target_crs = graph.graph['crs']
        gdf_green, gdf_water = get_green_and_water_areas(central_point, distance, target_crs)
        gdf_pois = _get_colored_pois(central_point, distance, target_crs)
        center_node = get_center_node(graph, central_point)

        for dataset in datasets:
            profile_key = dataset['profile_key']
            print(f"\n--- Profile: {profile_key} ---")
            df_walkability = pd.read_csv(dataset['file_path'])
            if 'IQC' not in df_walkability.columns:
                print("Dataset has no IQC column. Skipping.")
                continue

            # Baseline heatmap (IQC as stored in the dataset).
            plot_walkability_heatmap(
                df_walkability=df_walkability,
                graph=graph,
                center_node=center_node,
                gdf_green=gdf_green,
                gdf_water=gdf_water,
                location=location,
                profile_key=profile_key,
                base_dir=dataset['base_dir'],
                distance=distance,
                title=f'Baseline IQC - {location} - {profile_key}',
                gdf_pois=gdf_pois,
                filename_suffix='_baseline',
            )

            if overlay_method is None:
                continue

            allocation_path = _find_latest_allocation(overlay_method, key_location, profile_key)
            if allocation_path is None:
                print(f"No {overlay_method} allocation found for this profile. Skipping overlay.")
                continue
            if dataset['hex_time_matrix_path'] is None:
                print("Dataset has no hex-time matrix; cannot recompute post-allocation IQC.")
                continue

            print(f"Using allocation: {allocation_path}")
            df_allocation = pd.read_csv(allocation_path)
            df_hex_time_matrix = pd.read_csv(dataset['hex_time_matrix_path'])

            df_after = _recompute_iqc_with_allocation(
                df_walkability=df_walkability,
                df_hex_time_matrix=df_hex_time_matrix,
                df_allocation=df_allocation,
            )
            gdf_installed = _installed_pois_gdf(df_allocation, target_crs)
            method_label = METHOD_LABELS.get(overlay_method, overlay_method.upper())

            plot_walkability_heatmap(
                df_walkability=df_after,
                graph=graph,
                center_node=center_node,
                gdf_green=gdf_green,
                gdf_water=gdf_water,
                location=location,
                profile_key=profile_key,
                base_dir=dataset['base_dir'],
                distance=distance,
                title=f'IQC after {method_label} allocation - {location} - {profile_key}',
                gdf_pois=gdf_pois,
                gdf_installed_pois=gdf_installed,
                installed_pois_label='Allocated POIs',
                filename_suffix=f'_after_{overlay_method}',
            )

    print("\nIQC map generation finished.")
