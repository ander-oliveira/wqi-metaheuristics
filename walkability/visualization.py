from .common import *

from .hexagons import h3_to_polygon, plot_h3_grid, _process_hex_polygon

def plot_basic_map(graph, center_node: int, gdf_green: gpd.GeoDataFrame, 
                   gdf_water: gpd.GeoDataFrame, 
                   title: str, filename: str) -> None:
    
    edges_gdf = ox.graph_to_gdfs(graph, nodes=False, fill_edge_geometry=True)
    
    fig, ax = plt.subplots(figsize=(14, 14), dpi=300, facecolor='white')
    ax.set_facecolor('white')
    
    # Create geographic boundary from graph edges
    map_boundary = edges_gdf.union_all().convex_hull

    if not gdf_green.empty:
        gdf_green_clipped = gpd.clip(gdf_green, map_boundary)
        if not gdf_green_clipped.empty:
            gdf_green_clipped.plot(ax=ax, color='green', alpha=0.5, zorder=1)
    
    if not gdf_water.empty:
        gdf_water_clipped = gpd.clip(gdf_water, map_boundary)
        if not gdf_water_clipped.empty:
            gdf_water_clipped.plot(ax=ax, color='blue', alpha=0.5, zorder=1)
    
    edges_gdf.plot(ax=ax, color="black", linewidth=0.5, alpha=0.8, zorder=3)

    center_coords = graph.nodes[center_node]
    ax.scatter(center_coords['x'], center_coords['y'], c='red', s=50, zorder=5)

    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                             markersize=8, label='Central Point')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
             frameon=True, fancybox=True, shadow=False)
    
    plt.title(title, fontsize=16)
    plt.axis("off")
    
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def plot_base_map(ax, graph, gdf_green: gpd.GeoDataFrame, 
                  gdf_water: gpd.GeoDataFrame, iso_colors: list,
                  iso_intervals: list, center_node: int) -> tuple:
    """
    Plots the base map with green areas, water, edges, and isochrones.
    Returns edges_gdf for further use.
    
    Args:
        ax: Matplotlib axis object
        graph: NetworkX MultiDiGraph with travel_time and edge_color attributes
        gdf_green: GeoDataFrame with green areas
        gdf_water: GeoDataFrame with water bodies
        iso_colors: List of colors for isochrone intervals
        iso_intervals: List of time intervals in minutes
        center_node: Origin node ID
        
    Returns:
        tuple: (edges_gdf, colored_edges, black_edges)
    """
    edges_gdf = ox.graph_to_gdfs(graph, nodes=False, fill_edge_geometry=True)
    colored_edges = edges_gdf[edges_gdf["travel_time"] <= iso_intervals[-1]]
    black_edges = edges_gdf[edges_gdf["travel_time"] > iso_intervals[-1]]
    
    # Create geographic boundary from graph edges
    map_boundary = edges_gdf.union_all().convex_hull
    
    # Plot green areas
    if not gdf_green.empty:
        gdf_green_clipped = gpd.clip(gdf_green, map_boundary)
        if not gdf_green_clipped.empty:
            gdf_green_clipped.plot(ax=ax, color='green', alpha=0.5, zorder=1)
    
    # Plot water bodies
    if not gdf_water.empty:
        gdf_water_clipped = gpd.clip(gdf_water, map_boundary)
        if not gdf_water_clipped.empty:
            gdf_water_clipped.plot(ax=ax, color='blue', alpha=0.5, zorder=1)

    # Plot colored edges (within isochrone)
    for color, group in colored_edges.groupby("edge_color"):
        group.plot(ax=ax, color=color, linewidth=2, alpha=0.8, zorder=3)

    # Plot black edges (outside isochrone)
    if not black_edges.empty:
        black_edges.plot(ax=ax, color="black", linewidth=0.5, alpha=0.5, zorder=3)
    
    # Plot center point
    center_coords = graph.nodes[center_node]
    ax.scatter(center_coords['x'], center_coords['y'], c='red', s=50, zorder=5)
    
    return edges_gdf, colored_edges, black_edges

def plot_map_with_isochrones(graph, center_node: int, iso_colors: list, 
                             iso_intervals: list, gdf_green: gpd.GeoDataFrame, 
                             gdf_water: gpd.GeoDataFrame, place: tuple, 
                             distance: float,
                             title: str, filename: str) -> None:
    """
    Plots a map with isochrones only (no POIs or H3 grid).
    """
    fig, ax = plt.subplots(figsize=(14, 14), dpi=300, facecolor='white')
    ax.set_facecolor('white')
    
    # Plot base map elements
    plot_base_map(ax, graph, gdf_green, gdf_water, iso_colors, iso_intervals, center_node)

    legend_elements_time = [
        Line2D([0], [0], color=iso_colors[0], lw=2, label="0 - 5 min"),
        Line2D([0], [0], color=iso_colors[1], lw=2, label="5 - 10 min"),
        Line2D([0], [0], color=iso_colors[2], lw=2, label="10 - 15 min"),
        Line2D([0], [0], color=iso_colors[3], lw=2, label="15 - 20 min"),
        Line2D([0], [0], color="black", lw=0.5, label="> 20 min")
    ]
    
    ax.legend(handles=legend_elements_time, loc="lower left", title="Access Time")
    
    plt.title(title, fontsize=16)
    plt.axis("off")
    
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def _add_poi_legend_below(fig, gdf_pois: gpd.GeoDataFrame, ncols: int = 4) -> None:
    """
    Add a horizontal POI legend below the map without overlapping the figure.
    Image height expands automatically via bbox_inches='tight'.
    """
    if gdf_pois.empty:
        return

    legend_elements_poi = []
    unique_pois = (
        gdf_pois[['poi_type', 'color']]
        .dropna(subset=['poi_type'])
        .drop_duplicates()
        .sort_values('poi_type')
    )

    for _, row in unique_pois.iterrows():
        poi_label = str(row['poi_type']) if pd.notna(row['poi_type']) else 'other'
        count = int((gdf_pois['poi_type'] == row['poi_type']).sum())
        label = f"{poi_label.replace('_', ' ').title()} ({count})"
        legend_elements_poi.append(
            Line2D([0], [0], marker='o', color='w',
                   label=label, markerfacecolor=row['color'],
                   markeredgecolor='black', markersize=8)
        )

    if not legend_elements_poi:
        return

    fig.legend(
        handles=legend_elements_poi,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.0),   # abaixo do eixo
        ncol=ncols,
        title='POIs',
        fontsize=9,
        title_fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=False,
        borderpad=0.8,
        columnspacing=1.2,
        handletextpad=0.5,
    )

def plot_map_with_pois_isochrones(graph, center_node: int, iso_colors: list,
                                  iso_intervals: list, gdf_green: gpd.GeoDataFrame,
                                  gdf_water: gpd.GeoDataFrame,
                                  gdf_pois: gpd.GeoDataFrame,
                                  title: str, filename: str) -> None:
    """
    Plots a map with isochrones and POIs (no H3 grid).
    POI legend is rendered below the map, horizontal, 4 columns.
    """
    fig, ax = plt.subplots(figsize=(14, 14), dpi=300, facecolor='white')
    ax.set_facecolor('white')

    plot_base_map(ax, graph, gdf_green, gdf_water, iso_colors, iso_intervals, center_node)

    if not gdf_pois.empty:
        for color, group in gdf_pois.groupby('color'):
            group.plot(ax=ax, color=color, marker='o', markersize=40,
                       edgecolor='black', zorder=6)

    # Isochrone legend - inside the map (lower-left)
    legend_elements_time = [
        Line2D([0], [0], color=iso_colors[0], lw=2, label="0 - 5 min"),
        Line2D([0], [0], color=iso_colors[1], lw=2, label="5 - 10 min"),
        Line2D([0], [0], color=iso_colors[2], lw=2, label="10 - 15 min"),
        Line2D([0], [0], color=iso_colors[3], lw=2, label="15 - 20 min"),
        Line2D([0], [0], color="black", lw=0.5, label="> 20 min"),
    ]
    ax.legend(handles=legend_elements_time, loc="lower left", title="Access Time")

    plt.title(title, fontsize=16)
    plt.axis("off")

    # POI legend - below the map
    _add_poi_legend_below(fig, gdf_pois)

    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def plot_map_with_all_pois(graph, center_node: int, iso_colors: list,
                           iso_intervals: list, gdf_green: gpd.GeoDataFrame,
                           gdf_water: gpd.GeoDataFrame,
                           gdf_pois: gpd.GeoDataFrame,
                           title: str, filename: str) -> None:
    """
    Plots a map with all POIs in the complete area (not filtered by accessibility).
    POI legend is rendered below the map, horizontal, 4 columns.
    """
    fig, ax = plt.subplots(figsize=(14, 14), dpi=300, facecolor='white')
    ax.set_facecolor('white')

    plot_base_map(ax, graph, gdf_green, gdf_water, iso_colors, iso_intervals, center_node)

    if not gdf_pois.empty:
        for color, group in gdf_pois.groupby('color'):
            group.plot(ax=ax, color=color, marker='o', markersize=40,
                       edgecolor='black', zorder=6)

    legend_elements_time = [
        Line2D([0], [0], color=iso_colors[0], lw=2, label="0 - 5 min"),
        Line2D([0], [0], color=iso_colors[1], lw=2, label="5 - 10 min"),
        Line2D([0], [0], color=iso_colors[2], lw=2, label="10 - 15 min"),
        Line2D([0], [0], color=iso_colors[3], lw=2, label="15 - 20 min"),
        Line2D([0], [0], color="black", lw=0.5, label="> 20 min"),
    ]
    ax.legend(handles=legend_elements_time, loc="lower left", title="Access Time")

    plt.title(title, fontsize=16)
    plt.axis("off")

    _add_poi_legend_below(fig, gdf_pois)

    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ Map with all POIs saved: {filename}")

def plot_map_with_crosswalks_signals(graph, center_node: int, iso_colors: list, 
                                     iso_intervals: list, gdf_green: gpd.GeoDataFrame, 
                                     gdf_water: gpd.GeoDataFrame, 
                                     gdf_crosswalks: gpd.GeoDataFrame,
                                     gdf_traffic_signals: gpd.GeoDataFrame,
                                     title: str, filename: str) -> None:
    """
    Plots a map with all crosswalks and traffic signals in the complete area.
    Uses the same configuration as the main map with edges, green areas, and water bodies.
    
    Args:
        graph: NetworkX MultiDiGraph
        center_node: Origin node ID
        iso_colors: List of colors for isochrone intervals
        iso_intervals: List of time intervals in minutes
        gdf_green: GeoDataFrame with green areas
        gdf_water: GeoDataFrame with water bodies
        gdf_crosswalks: GeoDataFrame with all crosswalks
        gdf_traffic_signals: GeoDataFrame with all traffic signals
        title: Map title
        filename: Output filename
    """
    fig, ax = plt.subplots(figsize=(14, 14), dpi=300, facecolor='white')
    ax.set_facecolor('white')
    
    # Plot base map elements (same configuration as main map)
    plot_base_map(ax, graph, gdf_green, gdf_water, iso_colors, iso_intervals, center_node)
    
    # Plot traffic signals (small orange circles, no border) — below crosswalks
    if not gdf_traffic_signals.empty:
        gdf_traffic_signals.plot(
            ax=ax, color='orange', marker='o', markersize=8,
            edgecolor='none', zorder=6
        )

    # Plot crosswalks (small blue X markers) — on top of signals
    if not gdf_crosswalks.empty:
        gdf_crosswalks.plot(
            ax=ax, color='blue', marker='x', markersize=14,
            linewidth=0.6, zorder=7
        )

    legend_elements_time = [
        Line2D([0], [0], color=iso_colors[0], lw=2, label="0 - 5 min"),
        Line2D([0], [0], color=iso_colors[1], lw=2, label="5 - 10 min"),
        Line2D([0], [0], color=iso_colors[2], lw=2, label="10 - 15 min"),
        Line2D([0], [0], color=iso_colors[3], lw=2, label="15 - 20 min"),
        Line2D([0], [0], color="black", lw=0.5, label="> 20 min")
    ]
    
    time_legend = ax.legend(
        handles=legend_elements_time, loc="lower left", title="Access Time"
    )
    ax.add_artist(time_legend)
    
    # Add legend for crosswalks and traffic signals
    legend_elements_features = []
    if not gdf_crosswalks.empty:
        legend_elements_features.append(
            Line2D([0], [0], marker='x', color='blue',
                   label=f'Crosswalks ({len(gdf_crosswalks)})', markerfacecolor='blue',
                   markeredgecolor='none', markersize=8, markeredgewidth=0.6,
                   linestyle='None')
        )
    if not gdf_traffic_signals.empty:
        legend_elements_features.append(
            Line2D([0], [0], marker='o', color='w',
                   label=f'Traffic Signals ({len(gdf_traffic_signals)})', markerfacecolor='orange',
                   markeredgecolor='none', markersize=6, markeredgewidth=0)
        )
    
    if legend_elements_features:
        ax.legend(
            handles=legend_elements_features, loc='upper right', 
            title="Features", fontsize=9
        )
    
    plt.title(title, fontsize=16)
    plt.axis("off")
    
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ Map with crosswalks and traffic signals saved: {filename}")

def plot_map_with_pois_isochrones_h3(graph, center_node: int, iso_colors: list, 
                                     iso_intervals: list, gdf_green: gpd.GeoDataFrame, 
                                     gdf_water: gpd.GeoDataFrame, 
                                     gdf_pois: gpd.GeoDataFrame,
                                     place: tuple, distance: float,
                                     title: str, filename: str,
                                     h3_resolution: int) -> None:
    """
    Plots a map with isochrones, POIs, and H3 hexagonal grid.
    """
    fig, ax = plt.subplots(figsize=(14, 14), dpi=300, facecolor='white')
    ax.set_facecolor('white')
    
    # Plot base map elements
    plot_base_map(ax, graph, gdf_green, gdf_water, iso_colors, iso_intervals, center_node)
    
    # Plot H3 grid overlay
    h3_plotted = plot_h3_grid(ax, graph, place, distance, h3_resolution)
    if h3_plotted == 0:
        print("⚠ H3 grid warning: no hexagons intersected the map boundary for plotting.")
    
    if not gdf_pois.empty:
        for color, group in gdf_pois.groupby('color'):
            group.plot(
                ax=ax, color=color, marker='o', markersize=40, 
                edgecolor='black', zorder=6
            )
    
    legend_elements_time = [
        Line2D([0], [0], color=iso_colors[0], lw=2, label="0 - 5 min"),
        Line2D([0], [0], color=iso_colors[1], lw=2, label="5 - 10 min"),
        Line2D([0], [0], color=iso_colors[2], lw=2, label="10 - 15 min"),
        Line2D([0], [0], color=iso_colors[3], lw=2, label="15 - 20 min"),
        Line2D([0], [0], color="black", lw=0.5, label="> 20 min"),
        Line2D([0], [0], color="#4d4d4d", lw=1.2, label=f"H3 Grid (res {h3_resolution})")
    ]
    
    time_legend = ax.legend(
        handles=legend_elements_time, loc="lower left", title="Access Time"
    )
    ax.add_artist(time_legend)
    
    plt.title(title, fontsize=16)
    plt.axis("off")

    _add_poi_legend_below(fig, gdf_pois)

    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n-> Map with POIs and isochrones saved as '{filename}'")

def plot_map_with_selected_hexagons(graph, center_node: int, iso_colors: list, 
                                    iso_intervals: list, gdf_green: gpd.GeoDataFrame, 
                                    gdf_water: gpd.GeoDataFrame, 
                                    gdf_pois: gpd.GeoDataFrame,
                                    place: tuple, distance: float,
                                    selected_hex_ids: list,
                                    title: str, filename: str,
                                    h3_resolution: int) -> None:
    """
    Plots a map with isochrones, POIs, H3 hexagonal grid, and highlights selected hexagons in fuschia.
    Similar to plot_map_with_pois_isochrones_h3 but with selected hexagons filled in fuschia color.
    Uses multiprocessing for faster hexagon polygon generation.
    
    Args:
        graph: NetworkX MultiDiGraph
        center_node: Origin node ID
        iso_colors: List of colors for isochrone intervals
        iso_intervals: List of time intervals in minutes
        gdf_green: GeoDataFrame with green areas
        gdf_water: GeoDataFrame with water bodies
        gdf_pois: GeoDataFrame with POIs
        place: Tuple with (latitude, longitude) of central point
        distance: Analysis radius in meters
        selected_hex_ids: List of H3 hexagon IDs to highlight
        title: Plot title
        filename: Output filename
        h3_resolution: H3 resolution level
    """
    fig, ax = plt.subplots(figsize=(14, 14), dpi=300, facecolor='white')
    ax.set_facecolor('white')
    
    # Plot base map elements
    plot_base_map(ax, graph, gdf_green, gdf_water, iso_colors, iso_intervals, center_node)
    
    # Get target CRS and map boundary
    target_crs = graph.graph['crs']
    gdf_nodes = ox.graph_to_gdfs(graph, edges=False)
    map_boundary = gdf_nodes.union_all().convex_hull
    
    # Calculate hexagon IDs that cover the analysis area
    hex_center = h3.latlng_to_cell(place[0], place[1], h3_resolution)
    hex_radius = math.ceil(distance / h3.average_hexagon_edge_length(h3_resolution, unit='m'))
    hex_ids = list(h3.grid_disk(hex_center, hex_radius))
    
    # Convert selected_hex_ids to set for faster lookup
    selected_set = set(selected_hex_ids)
    
    # Prepare multiprocessing for polygon generation
    total_hexagons = len(hex_ids)
    print(f"\nGenerating {total_hexagons} hexagon polygons for visualization...")
    
    # Use multiprocessing for faster polygon generation
    use_multiprocessing = total_hexagons > 100
    
    if use_multiprocessing:
        print(f"  Using multiprocessing with {cpu_count()} CPU cores...")
        map_boundary_wkt = map_boundary.wkt
        
        # Prepare arguments
        args_list = [
            (hex_id, target_crs, map_boundary_wkt, hex_id in selected_set)
            for hex_id in hex_ids
        ]
        
        # Process in parallel
        num_processes = max(1, cpu_count() - 1)
        with Pool(processes=num_processes) as pool:
            results = pool.map(_process_hex_polygon, args_list)
        
        # Filter out None results
        hexagon_polygons = [r for r in results if r is not None]
        print(f"  Completed: {len(hexagon_polygons)}/{total_hexagons} hexagons processed")
    else:
        print("  Using serial processing...")
        hexagon_polygons = []
        for hex_id in hex_ids:
            hex_poly = h3_to_polygon(hex_id, target_crs=target_crs)
            if hex_poly.intersects(map_boundary):
                is_selected = hex_id in selected_set
                hexagon_polygons.append((hex_id, hex_poly, is_selected))
    
    # Plot hexagons
    print(f"  Plotting {len(hexagon_polygons)} hexagons...")
    for hex_id, hex_poly, is_selected in hexagon_polygons:
        if is_selected:
            # Fill selected hexagons with fuschia
            patch = plt.Polygon(
                list(hex_poly.exterior.coords), 
                edgecolor='grey', 
                facecolor='fuchsia',
                linewidth=0.4, 
                alpha=0.6, 
                zorder=4
            )
        else:
            # Regular hexagons without fill
            patch = plt.Polygon(
                list(hex_poly.exterior.coords), 
                edgecolor='grey', 
                facecolor='none',
                linewidth=0.4, 
                alpha=0.7, 
                zorder=2
            )
        ax.add_patch(patch)
    
    # Plot POIs
    if not gdf_pois.empty:
        for color, group in gdf_pois.groupby('color'):
            group.plot(
                ax=ax, color=color, marker='o', markersize=40, 
                edgecolor='black', zorder=6
            )
    
    # Create legends
    legend_elements_time = [
        Line2D([0], [0], color=iso_colors[0], lw=2, label="0 - 5 min"),
        Line2D([0], [0], color=iso_colors[1], lw=2, label="5 - 10 min"),
        Line2D([0], [0], color=iso_colors[2], lw=2, label="10 - 15 min"),
        Line2D([0], [0], color=iso_colors[3], lw=2, label="15 - 20 min"),
        Line2D([0], [0], color="black", lw=0.5, label="> 20 min"),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='fuchsia', 
               markersize=10, label=f'Selected Hexagons ({len(selected_hex_ids)})', alpha=0.6)
    ]
    
    time_legend = ax.legend(
        handles=legend_elements_time, loc="lower left", title="Access Time & Hexagons"
    )
    ax.add_artist(time_legend)
    
    plt.title(title, fontsize=16)
    plt.axis("off")

    _add_poi_legend_below(fig, gdf_pois)

    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n-> Map with selected hexagons saved as '{filename}'")

def plot_walkability_heatmap(df_walkability: pd.DataFrame,
                             graph: nx.MultiDiGraph,
                             center_node: int,
                             gdf_green: gpd.GeoDataFrame,
                             gdf_water: gpd.GeoDataFrame,
                             location: str,
                             profile_key: str,
                             base_dir: str = 'data',
                             distance: int = None,
                             title: str = None,
                             gdf_pois: gpd.GeoDataFrame = None,
                             gdf_installed_pois: gpd.GeoDataFrame = None,
                             installed_pois_label: str = 'POIs instalados',
                             filename_suffix: str = '') -> None:
    """
    Generates a heatmap visualization of the walkability index (IQC).

    Hexagons are colored according to their IQC value (0-1 scale)
    using a heatmap color palette. The map includes green/water areas,
    black edges, and a vertical colorbar legend.

    Args:
        df_walkability: DataFrame with h3_id and IQC columns [0, 1]
        graph: NetworkX MultiDiGraph with the street network
        center_node: Origin node ID
        gdf_green: GeoDataFrame with green areas
        gdf_water: GeoDataFrame with water bodies
        location: Location name for saving the output
        profile_key: Profile key for file naming
        base_dir: Base directory for data storage
        title: Optional title shown above the map.
        gdf_pois: Optional GeoDataFrame with existing POIs. If present,
            POIs are plotted with their color column and a legend below the map.
        gdf_installed_pois: Optional GeoDataFrame with installed POIs
            (geometry + 'quantity' column). Plotted as black dots sized by
            quantity, with a legend entry.
        installed_pois_label: Legend label for the installed POIs.
        filename_suffix: Optional suffix appended to output file names
            (e.g. '_baseline', '_after_grasp').
    """
    print(f"\n{'='*60}")
    print(f"GENERATING WALKABILITY HEATMAP")
    print(f"{'='*60}")
    
    # IQC is already in [0, 1] scale
    iqc_min = df_walkability['IQC'].min()
    iqc_max = df_walkability['IQC'].max()
    
    # Create hexagon geometries from h3_id
    hexagons = []
    for _, row in df_walkability.iterrows():
        h3_boundary = h3.cell_to_boundary(row['h3_id'])
        # H3 returns (lat, lon), need to convert to (lon, lat) for Shapely
        polygon = Polygon([(lon, lat) for lat, lon in h3_boundary])
        hexagons.append({
            'h3_id': row['h3_id'],
            'IQC': row['IQC'],
            'geometry': polygon
        })
    
    # Create GeoDataFrame with hexagons
    gdf_hexagons = gpd.GeoDataFrame(hexagons, crs='EPSG:4326')
    
    # Convert to same CRS as graph
    target_crs = graph.graph['crs']
    gdf_hexagons = gdf_hexagons.to_crs(target_crs)
    
    # Get edges for boundary and plotting
    edges_gdf = ox.graph_to_gdfs(graph, nodes=False, fill_edge_geometry=True)
    map_boundary = edges_gdf.union_all().convex_hull
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Plot green areas
    if not gdf_green.empty:
        gdf_green_clipped = gpd.clip(gdf_green, map_boundary)
        if not gdf_green_clipped.empty:
            gdf_green_clipped.plot(ax=ax, color='green', alpha=0.5, zorder=1)
    
    # Plot water bodies
    if not gdf_water.empty:
        gdf_water_clipped = gpd.clip(gdf_water, map_boundary)
        if not gdf_water_clipped.empty:
            gdf_water_clipped.plot(ax=ax, color='blue', alpha=0.5, zorder=1)
    
    # Plot hexagons with heatmap colors
    # Use a colormap (hot, plasma, inferno, viridis, etc.)
    cmap = plt.cm.RdBu_r  # Reversed Red-Blue: blue (low) -> white -> red (high)
    # Alternative colormaps: plt.cm.plasma, plt.cm.viridis, plt.cm.YlOrRd, plt.cm.hot_r
    
    gdf_hexagons.plot(
        ax=ax,
        column='IQC',
        cmap=cmap,
        edgecolor='face',
        linewidth=0.5,
        alpha=0.8,
        vmin=0,
        vmax=1,
        zorder=4
    )
    
    # Plot edges in black
    edges_gdf.plot(ax=ax, color='black', linewidth=0.5, alpha=0.5, zorder=3)

    # Plot existing POIs, when available.
    has_pois = gdf_pois is not None and not gdf_pois.empty
    if has_pois:
        gdf_pois_clipped = gpd.clip(gdf_pois, map_boundary)
        if not gdf_pois_clipped.empty:
            if 'color' not in gdf_pois_clipped.columns:
                gdf_pois_clipped = gdf_pois_clipped.copy()
                gdf_pois_clipped['color'] = 'grey'
            for color, group in gdf_pois_clipped.groupby('color'):
                group.plot(
                    ax=ax,
                    color=color,
                    marker='o',
                    markersize=20,
                    edgecolor='black',
                    linewidth=0.3,
                    alpha=0.9,
                    zorder=5,
                )
            gdf_pois_for_legend = gdf_pois_clipped
        else:
            gdf_pois_for_legend = gdf_pois
    else:
        gdf_pois_for_legend = None
    
    # Plot center point
    center_coords = graph.nodes[center_node]
    ax.scatter(center_coords['x'], center_coords['y'], c='red', s=50, zorder=6)

    # Plot installed POIs (black dots sized by quantity)
    has_installed_pois = gdf_installed_pois is not None and not gdf_installed_pois.empty
    if has_installed_pois:
        quantities = gdf_installed_pois.get('quantity', pd.Series(1, index=gdf_installed_pois.index))
        sizes = 14 + 10 * pd.to_numeric(quantities, errors='coerce').fillna(1).clip(lower=1)
        ax.scatter(
            gdf_installed_pois.geometry.x, gdf_installed_pois.geometry.y,
            c='black', s=sizes, edgecolor='white', linewidth=0.5, zorder=7,
        )
        ax.legend(
            handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                            markeredgecolor='white', markersize=8,
                            label=installed_pois_label)],
            loc='upper right', fontsize=10, frameon=True, fancybox=True,
        )

    # Add colorbar (vertical, on the right)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Índice de Caminhabilidade (IQC)', fontsize=12, weight='bold')
    
    # Set colorbar ticks for [0, 1] scale
    tick_positions = np.linspace(0, 1, 6)  # 6 ticks: 0, 0.2, 0.4, 0.6, 0.8, 1.0
    tick_labels = [f'{pos:.2f}' for pos in tick_positions]
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)
    
    # Remove axis
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=16, weight='bold', pad=14)

    if gdf_pois_for_legend is not None:
        _add_poi_legend_below(fig, gdf_pois_for_legend)
    
    # Save figure
    output_dir = f"{base_dir}/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    dist_suffix = f"_dist{distance}" if distance is not None else ""
    output_file = f"{output_dir}/{location}_walkability_heatmap_{profile_key}{dist_suffix}{filename_suffix}.png"
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Walkability heatmap saved: {output_file}")
    print(f"  Total hexagons plotted: {len(gdf_hexagons)}")
    print(f"  IQC range: [{iqc_min:.4f}, {iqc_max:.4f}]")
    
    # ====================================================================
    # CLASSIFIED CHOROPLETH MAP (Natural Breaks - Jenks)
    # ====================================================================
    # Only generate if there are enough distinct values for classification
    n_unique = gdf_hexagons['IQC'].nunique()
    if n_unique >= 5:
        n_classes = min(5, n_unique)  # 5 classes or fewer if not enough unique values
        
        # Available schemes: 'NaturalBreaks' (Jenks), 'Quantiles', 'EqualInterval',
        #                    'FisherJenks', 'StdMean', 'MaximumBreaks'
        classification_schemes = {
            'NaturalBreaks': 'natural_breaks',
            'Quantiles': 'quantiles',
            'EqualInterval': 'equal_interval'
        }
        
        for scheme_name, file_suffix in classification_schemes.items():
            try:
                fig_cls, ax_cls = plt.subplots(figsize=(12, 12))
                ax_cls.set_facecolor('white')
                fig_cls.patch.set_facecolor('white')
                
                # Plot green areas
                if not gdf_green.empty:
                    gdf_green_clipped = gpd.clip(gdf_green, map_boundary)
                    if not gdf_green_clipped.empty:
                        gdf_green_clipped.plot(ax=ax_cls, color='green', alpha=0.5, zorder=1)
                
                # Plot water bodies
                if not gdf_water.empty:
                    gdf_water_clipped = gpd.clip(gdf_water, map_boundary)
                    if not gdf_water_clipped.empty:
                        gdf_water_clipped.plot(ax=ax_cls, color='blue', alpha=0.5, zorder=1)
                
                # Plot classified hexagons
                gdf_hexagons.plot(
                    ax=ax_cls,
                    column='IQC',
                    scheme=scheme_name,
                    k=n_classes,
                    cmap=cmap,
                    edgecolor='black',
                    linewidth=0.3,
                    alpha=0.8,
                    legend=True,
                    legend_kwds={
                        'loc': 'lower right',
                        'fontsize': 9,
                        'title': f'IQC ({scheme_name})',
                        'title_fontsize': 10,
                        'frameon': True,
                        'framealpha': 0.9
                    },
                    zorder=4
                )
                
                # Plot edges
                edges_gdf.plot(ax=ax_cls, color='black', linewidth=0.5, alpha=0.5, zorder=3)

                # Plot existing POIs.
                if has_pois:
                    gdf_pois_clipped = gpd.clip(gdf_pois, map_boundary)
                    if not gdf_pois_clipped.empty:
                        if 'color' not in gdf_pois_clipped.columns:
                            gdf_pois_clipped = gdf_pois_clipped.copy()
                            gdf_pois_clipped['color'] = 'grey'
                        for color, group in gdf_pois_clipped.groupby('color'):
                            group.plot(
                                ax=ax_cls,
                                color=color,
                                marker='o',
                                markersize=20,
                                edgecolor='black',
                                linewidth=0.3,
                                alpha=0.9,
                                zorder=5,
                            )
                        gdf_pois_for_cls_legend = gdf_pois_clipped
                    else:
                        gdf_pois_for_cls_legend = gdf_pois
                else:
                    gdf_pois_for_cls_legend = None

                # Plot center point
                ax_cls.scatter(center_coords['x'], center_coords['y'], c='red', s=50, zorder=6)

                # Plot installed POIs, keeping the classification legend visible
                if has_installed_pois:
                    quantities = gdf_installed_pois.get('quantity', pd.Series(1, index=gdf_installed_pois.index))
                    sizes = 14 + 10 * pd.to_numeric(quantities, errors='coerce').fillna(1).clip(lower=1)
                    ax_cls.scatter(
                        gdf_installed_pois.geometry.x, gdf_installed_pois.geometry.y,
                        c='black', s=sizes, edgecolor='white', linewidth=0.5, zorder=7,
                    )
                    class_legend = ax_cls.get_legend()
                    ax_cls.legend(
                        handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                                        markeredgecolor='white', markersize=8,
                                        label=installed_pois_label)],
                        loc='upper right', fontsize=9, frameon=True, fancybox=True,
                    )
                    if class_legend is not None:
                        ax_cls.add_artist(class_legend)

                ax_cls.set_axis_off()
                if title:
                    ax_cls.set_title(title, fontsize=16, weight='bold', pad=14)

                if gdf_pois_for_cls_legend is not None:
                    _add_poi_legend_below(fig_cls, gdf_pois_for_cls_legend)

                cls_output = f"{output_dir}/{location}_walkability_heatmap_{profile_key}{dist_suffix}{filename_suffix}_{file_suffix}.png"
                plt.tight_layout()
                plt.savefig(cls_output, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig_cls)
                
                print(f"✓ Classified heatmap ({scheme_name}, k={n_classes}) saved: {cls_output}")
                
            except Exception as e:
                print(f"⚠ Could not generate {scheme_name} map: {e}")
                plt.close('all')
    else:
        print(f"⚠ Skipping classified maps: only {n_unique} distinct IQC values (need ≥ 5)")

