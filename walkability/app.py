from .common import *
from metaheuristics import ask_metaheuristic_method, load_seeds, walk_meta_opt

from .hexagons import select_random_hexagons
from .iqc import compute_walkability_for_all_hexagons
from .meta_inputs import build_hex_time_matrix
from .persistence import save_core_dataframes
from .pipeline import process_single_hexagon_optimized, run_analysis_pipeline
from .utils import build_analysis_base_dir, ensure_data_directories, select_locations
from .visualization import (
    plot_map_with_all_pois,
    plot_map_with_crosswalks_signals,
    plot_map_with_selected_hexagons,
    plot_walkability_heatmap,
)


def _ask_execution_mode() -> str:
    while True:
        print("\nSelect execution mode:")
        print("1 - Run full pipeline for all walking profiles")
        print("2 - Use existing dataset for selected location")
        choice = input("Enter option number: ").strip()
        if choice in {'1', '2'}:
            return choice
        print("Invalid option. Choose 1 or 2.")


def _build_profiles_to_run(walking_profiles: dict) -> list:
    profile_execution_order = ['average_adult', 'elderly', 'athlete']
    profiles_to_run = []
    for key in profile_execution_order:
        if key in walking_profiles:
            profiles_to_run.append((key, walking_profiles[key]))

    configured_keys = {key for key, _ in profiles_to_run}
    for key, profile_cfg in walking_profiles.items():
        if key not in configured_keys:
            profiles_to_run.append((key, profile_cfg))

    return profiles_to_run


def _find_existing_walkability_datasets(location: str,
                                        key_location: str,
                                        profile_keys: list,
                                        h3_resolution: int,
                                        distance: int) -> list:
    available = []
    suffix_template = f"_res_{h3_resolution}_dist{distance}.csv"

    for profile_key in profile_keys:
        base_dir = build_analysis_base_dir(key_location, profile_key, h3_resolution, distance)
        walkability_dir = os.path.join(base_dir, 'csv', 'walkability_index')
        if not os.path.isdir(walkability_dir):
            continue

        expected_suffix = f"_walkability_index_{profile_key}{suffix_template}"
        profile_files = []
        for file_name in sorted(os.listdir(walkability_dir)):
            if not file_name.endswith(expected_suffix):
                continue
            full_path = os.path.join(walkability_dir, file_name)
            if os.path.isfile(full_path):
                profile_files.append(full_path)

        if not profile_files:
            expected_file = os.path.join(
                walkability_dir,
                f"{location}_walkability_index_{profile_key}{suffix_template}"
            )
            if os.path.isfile(expected_file):
                profile_files.append(expected_file)

        for file_path in profile_files:
            file_name = os.path.basename(file_path)
            matrix_file_name = file_name.replace('_walkability_index_', '_hex_time_matrix_')
            matrix_file_path = os.path.join(walkability_dir, matrix_file_name)
            available.append({
                'profile_key': profile_key,
                'file_path': file_path,
                'hex_time_matrix_path': matrix_file_path,
                'has_hex_time_matrix': os.path.isfile(matrix_file_path),
            })

    return available


def _use_existing_dataset_mode(location: str,
                               key_location: str,
                               profiles_to_run: list,
                               h3_resolution: int,
                               distance: int) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
    profile_keys = [key for key, _ in profiles_to_run]
    available = _find_existing_walkability_datasets(
        location=location,
        key_location=key_location,
        profile_keys=profile_keys,
        h3_resolution=h3_resolution,
        distance=distance,
    )

    if not available:
        print("\nNo existing walkability datasets were found for this location.")
        return None, None

    print("\nExisting walkability datasets:")
    for idx, item in enumerate(available, start=1):
        matrix_status = 'ok' if item['has_hex_time_matrix'] else 'missing'
        print(
            f"{idx} - profile={item['profile_key']} | file={item['file_path']} "
            f"| hex_time_matrix={matrix_status}"
        )

    while True:
        choice = input("\nSelect a dataset to load [number] or press Enter to exit: ").strip()
        if choice == '':
            print("Dataset check finished.")
            return None, None, None
        if choice.isdigit():
            selected_idx = int(choice)
            if 1 <= selected_idx <= len(available):
                selected = available[selected_idx - 1]
                if not selected['has_hex_time_matrix']:
                    print("\nSelected dataset has no hex-time matrix file.")
                    print("Please run execution mode 1 again to generate both required inputs.")
                    return None, None, None

                df_selected = pd.read_csv(selected['file_path'])
                df_hex_time_matrix = pd.read_csv(selected['hex_time_matrix_path'])
                print(f"\nLoaded dataset for profile: {selected['profile_key']}")
                print(f"Rows: {len(df_selected)} | Columns: {len(df_selected.columns)}")
                print(f"Columns: {', '.join(df_selected.columns)}")
                print(f"Hex-time matrix rows: {len(df_hex_time_matrix)}")
                return df_selected, df_hex_time_matrix, selected['profile_key']
        print(f"Please enter a number between 1 and {len(available)}, or press Enter to exit.")


def _select_profile_for_meta(df_walkability_by_profile: dict) -> Tuple[str, pd.DataFrame]:
    if not df_walkability_by_profile:
        raise ValueError("No walkability datasets available.")

    profile_keys = list(df_walkability_by_profile.keys())
    if len(profile_keys) == 1:
        profile_key = profile_keys[0]
        return profile_key, df_walkability_by_profile[profile_key]

    print("\nSelect profile dataset for metaheuristic:")
    for idx, profile_key in enumerate(profile_keys, start=1):
        df_profile = df_walkability_by_profile[profile_key]
        print(f"{idx} - {profile_key} | rows={len(df_profile)}")

    while True:
        choice = input("Enter profile option number: ").strip()
        if choice.isdigit():
            selected_idx = int(choice)
            if 1 <= selected_idx <= len(profile_keys):
                selected_key = profile_keys[selected_idx - 1]
                return selected_key, df_walkability_by_profile[selected_key]
        print(f"Please enter a number between 1 and {len(profile_keys)}.")


def _run_metaheuristic_stage(df_walkability: pd.DataFrame,
                             df_hex_time_matrix: pd.DataFrame,
                             walking_profile: str,
                             budget: int,
                             location: str = None,
                             key_location: str = None,
                             h3_resolution: int = None,
                             distance: int = None) -> Optional[dict]:
    try:
        method = ask_metaheuristic_method()
        seeds = load_seeds('seeds.txt')
        return walk_meta_opt(
            df_walkability=df_walkability,
            df_hex_time_matrix=df_hex_time_matrix,
            budget=budget,
            method=method,
            seeds=seeds,
            walking_profile=walking_profile,
            location=location,
            key_location=key_location,
            h3_resolution=h3_resolution,
            distance=distance,
        )
    except Exception as e:
        print(f"Could not initialize metaheuristic stage: {e}")
        return None


def run_cli() -> None:
    def ask_yes_no(prompt: str) -> bool:
        while True:
            value = input(prompt).strip().lower()
            if value in {'y', 'yes'}:
                return True
            if value in {'n', 'no'}:
                return False
            print("Please enter 'y' or 'n'.")

    DISTANCE = 2000
    BUDGET = 100

    NETWORK_TYPE = 'walk'
    ISO_INTERVALS = [5, 10, 15, 20]
    H3_RESOLUTION = 9
    USE_PARALLEL_HEXAGON_PROCESSING = True
    # Master switch for all map plotting outputs.
    PLOT_ALL_MAPS = False
    GENERATE_WALKABILITY_HEATMAP = False
    SAVE_ONLY_DF_WALKABILITY = True
    SAVE_CRITIC_WEIGHTS_CSV = not SAVE_ONLY_DF_WALKABILITY
    SAVE_INDICATORS_BASE_CSV = not SAVE_ONLY_DF_WALKABILITY
    SAVE_HEX_TIME_MATRIX_CSV = True
    ISO_COLORS = ox.plot.get_colors(n=len(ISO_INTERVALS), cmap='plasma', start=0)

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

    WALKING_PROFILES = {
        'average_adult': {
            'name': 'Average Adult / Tourist',
            'speed_walk': 5.0,
            'uphill_factor': 0.80,
            'downhill_factor': 1.15,
        },
        'elderly': {
            'name': 'Elderly / Reduced Mobility',
            'speed_walk': 4.0,
            'uphill_factor': 0.65,
            'downhill_factor': 0.95,
        },
        'athlete': {
            'name': 'Young / Athlete',
            'speed_walk': 6.0,
            'uphill_factor': 0.90,
            'downhill_factor': 1.20,
        },
    }

    profiles_to_run = _build_profiles_to_run(WALKING_PROFILES)
    if not profiles_to_run:
        raise ValueError('No walking profiles configured.')

    execution_mode = _ask_execution_mode()
    selected_locations = select_locations(allow_all=(execution_mode == '1'))
    if not selected_locations:
        print("No locations selected.")
        return

    if execution_mode == '2':
        for location_idx, (_, location, _, key_location) in enumerate(selected_locations, start=1):
            if len(selected_locations) > 1:
                print(f"\n=== Location [{location_idx}/{len(selected_locations)}]: {location} ===")

            existing_df, existing_hex_time_matrix, existing_profile = _use_existing_dataset_mode(
                location=location,
                key_location=key_location,
                profiles_to_run=profiles_to_run,
                h3_resolution=H3_RESOLUTION,
                distance=DISTANCE,
            )
            if existing_df is None or existing_hex_time_matrix is None or existing_profile is None:
                print(f"Skipping location without selected dataset: {location}")
                continue

            _run_metaheuristic_stage(
                existing_df, existing_hex_time_matrix, existing_profile, BUDGET,
                location=location,
                key_location=key_location,
                h3_resolution=H3_RESOLUTION,
                distance=DISTANCE,
            )

        print('Done.')
        return

    for location_idx, (central_point, location, dem_path, key_location) in enumerate(selected_locations, start=1):
        if len(selected_locations) > 1:
            print(f"\n=== Location [{location_idx}/{len(selected_locations)}]: {location} ===")

        profile_names = ', '.join(key for key, _ in profiles_to_run)
        print(f'Running: {location} | distance={DISTANCE}m | profiles={profile_names}')

        shared_graph = None
        shared_green_water = None
        shared_raw_features = None
        df_walkability_by_profile = {}

        for profile_idx, (profile_key, selected_profile) in enumerate(profiles_to_run, start=1):
            analysis_base_dir = build_analysis_base_dir(key_location, profile_key, H3_RESOLUTION, DISTANCE)
            ensure_data_directories(analysis_base_dir)
            print(f"\n[{profile_idx}/{len(profiles_to_run)}] Profile: {profile_key}")

            walkability_indicators = []

            results = run_analysis_pipeline(
                central_point=central_point,
                location=location,
                dem_path=dem_path,
                profile=selected_profile,
                profile_key=profile_key,
                iso_intervals=ISO_INTERVALS,
                iso_colors=ISO_COLORS,
                poi_colors=POI_COLORS,
                distance=DISTANCE,
                h3_resolution=H3_RESOLUTION,
                network_type=NETWORK_TYPE,
                base_dir=analysis_base_dir,
                generate_h3=True,
                generate_visualizations=PLOT_ALL_MAPS,
                reuse_graph=shared_graph,
                reuse_green_water=shared_green_water,
                reuse_raw_features=shared_raw_features,
                force_recompute_tobler=(shared_graph is not None),
            )

            if shared_graph is None:
                shared_graph = results['graph']
                shared_green_water = (results['gdf_green'], results['gdf_water'])
                shared_raw_features = results['raw_features']

            if PLOT_ALL_MAPS and not results['raw_features']['pois'].empty:
                plot_map_with_all_pois(
                    results['graph'],
                    results['center_node'],
                    ISO_COLORS,
                    ISO_INTERVALS,
                    results['gdf_green'],
                    results['gdf_water'],
                    results['raw_features']['pois'],
                    title=f'Complete Area - All POIs - {location}',
                    filename=f'{analysis_base_dir}/visualizations/map_all_pois_{location}_{profile_key}_res{H3_RESOLUTION}_dist{DISTANCE}',
                )

            if PLOT_ALL_MAPS and (not results['raw_features']['crosswalks'].empty or not results['raw_features']['traffic_signals'].empty):
                plot_map_with_crosswalks_signals(
                    results['graph'],
                    results['center_node'],
                    ISO_COLORS,
                    ISO_INTERVALS,
                    results['gdf_green'],
                    results['gdf_water'],
                    results['raw_features']['crosswalks'],
                    results['raw_features']['traffic_signals'],
                    title=f'Complete Area - Crosswalks & Traffic Signals - {location}',
                    filename=f'{analysis_base_dir}/visualizations/map_crosswalks_signals_{location}_{profile_key}_res{H3_RESOLUTION}_dist{DISTANCE}',
                )

            initial_hex_df = results.get('df_hexagons', pd.DataFrame())

            selected_hex_ids = select_random_hexagons(initial_hex_df)

            generate_hex_visualizations = False
            hexagons_for_visualization = []
            if PLOT_ALL_MAPS and selected_hex_ids:
                generate_hex_visualizations = ask_yes_no('Generate visualization maps for selected hexagons? (y/n): ')
                if generate_hex_visualizations:
                    hexagons_for_visualization = selected_hex_ids.copy()

            plot_selected_hex_map = False
            if PLOT_ALL_MAPS and selected_hex_ids:
                plot_selected_hex_map = ask_yes_no('Plot map with selected hexagons highlighted? (y/n): ')

            if PLOT_ALL_MAPS and selected_hex_ids and plot_selected_hex_map:
                plot_map_with_selected_hexagons(
                    results['graph'],
                    results['center_node'],
                    ISO_COLORS,
                    ISO_INTERVALS,
                    results['gdf_green'],
                    results['gdf_water'],
                    results['accessible_features']['pois'],
                    central_point,
                    DISTANCE,
                    selected_hex_ids,
                    title=f'Map with Selected Hexagons ({len(selected_hex_ids)}) - {location}',
                    filename=f'{analysis_base_dir}/visualizations/map_selected_hexagons_{location}_{profile_key}_dist{DISTANCE}',
                    h3_resolution=H3_RESOLUTION,
                )

            if selected_hex_ids:
                num_cores = max(1, cpu_count() - 1)
                args_list = [
                    (
                        i + 1,
                        hex_id,
                        results['graph'],
                        results['gdf_green'],
                        results['gdf_water'],
                        results['raw_features'],
                        dem_path,
                        selected_profile,
                        profile_key,
                        ISO_INTERVALS,
                        ISO_COLORS,
                        POI_COLORS,
                        DISTANCE,
                        H3_RESOLUTION,
                        NETWORK_TYPE,
                        generate_hex_visualizations and (hex_id in hexagons_for_visualization),
                        analysis_base_dir,
                    )
                    for i, hex_id in enumerate(selected_hex_ids)
                ]

                if USE_PARALLEL_HEXAGON_PROCESSING and len(selected_hex_ids) > 1:
                    with ProcessPoolExecutor(max_workers=num_cores) as executor:
                        for idx, hex_id, indicators, error in executor.map(process_single_hexagon_optimized, args_list):
                            if error:
                                print(f'Error [{idx}/{len(selected_hex_ids)}] {hex_id}: {error}')
                                continue
                            walkability_indicators.append(indicators)
                else:
                    for args in args_list:
                        idx, hex_id, indicators, error = process_single_hexagon_optimized(args)
                        if error:
                            print(f'Error [{idx}/{len(selected_hex_ids)}] {hex_id}: {error}')
                            continue
                        walkability_indicators.append(indicators)

            if walkability_indicators:
                df_indicators_base = pd.DataFrame(walkability_indicators)
                if 'h3_id' in df_indicators_base.columns:
                    before_dedup = len(df_indicators_base)
                    df_indicators_base = df_indicators_base.drop_duplicates(subset=['h3_id'], keep='first')
                    duplicates_removed = before_dedup - len(df_indicators_base)
                    if duplicates_removed > 0:
                        print(f"Removed {duplicates_removed} duplicated hexagon rows by h3_id.")
                cols_order = [
                    'h3_id', 'latitude', 'longitude',
                    'S_saude', 'S_educacao', 'S_abastecimento', 'S_lazer', 'S_servicos',
                    'I_seguranca', 'A_vegetacao', 'A_agua', 'C_conectividade',
                    'T_transporte', 'U_urbanidade',
                ]
                df_indicators_base = df_indicators_base[cols_order]
            else:
                df_indicators_base = None
                print('No indicators collected.')

            df_walkability, critic_weights = compute_walkability_for_all_hexagons(
                location=location,
                profile_key=profile_key,
                h3_resolution=H3_RESOLUTION,
                base_dir=analysis_base_dir,
                df_indicators=df_indicators_base,
                distance=DISTANCE,
            )

            if df_indicators_base is not None and not df_indicators_base.empty:
                try:
                    df_hex_time_matrix = build_hex_time_matrix(
                        graph=results['graph'],
                        df_walkability_base=df_indicators_base,
                        profile=selected_profile,
                        profile_key=profile_key,
                        max_time=ISO_INTERVALS[-1],
                    )
                except Exception as e:
                    print(f"Could not build hex-time matrix for profile {profile_key}: {e}")
                    df_hex_time_matrix = pd.DataFrame()
            else:
                df_hex_time_matrix = pd.DataFrame()

            save_core_dataframes(
                df_walkability=df_walkability,
                critic_weights=critic_weights,
                df_indicators_base=df_indicators_base,
                df_hex_time_matrix=df_hex_time_matrix,
                location=location,
                profile_key=profile_key,
                h3_resolution=H3_RESOLUTION,
                base_dir=analysis_base_dir,
                distance=DISTANCE,
                save_walkability=True,
                save_critic_weights=SAVE_CRITIC_WEIGHTS_CSV,
                save_indicators_base=SAVE_INDICATORS_BASE_CSV,
                save_hex_time_matrix=SAVE_HEX_TIME_MATRIX_CSV,
            )

            if df_walkability is not None and not df_walkability.empty:
                df_walkability_by_profile[profile_key] = df_walkability

            if PLOT_ALL_MAPS and GENERATE_WALKABILITY_HEATMAP and not df_walkability.empty:
                plot_walkability_heatmap(
                    df_walkability=df_walkability,
                    graph=results['graph'],
                    center_node=results['center_node'],
                    gdf_green=results['gdf_green'],
                    gdf_water=results['gdf_water'],
                    location=location,
                    profile_key=profile_key,
                    base_dir=analysis_base_dir,
                    distance=DISTANCE,
                )

        if not df_walkability_by_profile:
            print("No walkability dataset was generated for this location.")
            continue

        print(
            "Walkability datasets generated successfully for this location. "
            "Use execution mode 2 to load a dataset and run metaheuristics."
        )

    print('Done.')
