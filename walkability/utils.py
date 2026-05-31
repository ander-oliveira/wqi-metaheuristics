from .common import *


def ensure_data_directories(base_dir: str = 'data') -> None:
    """
    Creates all necessary standard directories for the project.

    Args:
        base_dir: Base directory for data storage (default: 'data')
    """
    directories = [
        f'{base_dir}/csv/walkability_index',
        f'{base_dir}/visualizations',
        f'{base_dir}/cache'
    ]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Directory created: {directory}")


def sanitize_path_component(value: str) -> str:
    """Sanitize path components to avoid invalid characters on Windows."""
    if not value:
        return 'unknown'

    sanitized = re.sub(r'[<>:"/\\|?*]', '_', str(value).strip())
    sanitized = re.sub(r'\s+', '_', sanitized)
    sanitized = sanitized.strip('._')

    return sanitized or 'unknown'


def build_analysis_base_dir(key_location: str, profile_key: str, h3_resolution: int,
                            distance: int, data_root: str = 'data') -> str:
    """Build base_dir as data/location/<key_location>/profile/resolution_y/<distance>."""
    key_location_part = sanitize_path_component(key_location)
    profile_part = sanitize_path_component(profile_key)
    resolution_part = f"resolution_{h3_resolution}"
    distance_part = sanitize_path_component(distance)
    return os.path.join(data_root, 'location', key_location_part, profile_part, resolution_part, distance_part)


def _load_location_entries() -> list:
    entries = []
    with open('data/csv/locations.csv', 'r', encoding='utf-8') as f:
        csv_reader = csv.DictReader(f, delimiter=';')
        for row in csv_reader:
            key_location = (row.get('key_location') or row.get('location') or '').strip()
            address = (row.get('address') or '').strip()
            location = (row.get('location') or key_location or '').strip()
            coordinates = (row.get('coordinates') or '').strip()
            dem_file = (row.get('dem_file') or '').strip()

            if not location or not coordinates:
                continue

            entries.append({
                'key_location': key_location,
                'address': address,
                'location': location,
                'coordinates': coordinates,
                'dem_file': dem_file
            })

    return entries


def _entry_to_location_tuple(entry: dict) -> Tuple[Tuple[float, float], str, Optional[str], str]:
    coords = entry['coordinates'].strip('()').split(',')
    central_point = (float(coords[0]), float(coords[1]))
    return (
        central_point,
        entry['location'],
        entry.get('dem_file'),
        entry.get('key_location', entry['location'])
    )


def select_locations(allow_all: bool = True, force_all: bool = False) -> list:
    entries = _load_location_entries()
    if not entries:
        print("[WARNING] No locations found in locations.csv.")
        return []

    if force_all:
        return [_entry_to_location_tuple(entry) for entry in entries]

    print("\nSelect a location:")
    if allow_all:
        print("0 - ALL LOCATIONS")
    for i, entry in enumerate(entries, 1):
        print(f"{i} - {entry['location'].upper()} - {entry['address']}")

    while True:
        if allow_all:
            choice = input("\nEnter option number (or 0 for all): ").strip().lower()
        else:
            choice = input("\nEnter option number: ").strip().lower()

        if allow_all and choice in {'0', 'all'}:
            return [_entry_to_location_tuple(entry) for entry in entries]

        if choice.isdigit():
            selected_idx = int(choice)
            if 1 <= selected_idx <= len(entries):
                return [_entry_to_location_tuple(entries[selected_idx - 1])]

            if allow_all:
                print(f"Please enter between 1-{len(entries)}, or 0 for all.")
            else:
                print(f"Please enter between 1-{len(entries)}.")
            continue

        print("Invalid number. Try again.")


def select_location(allow_all: bool = False) -> Tuple[Tuple[float, float], str, Optional[str], str]:
    """Backward-compatible single-location selector."""
    selected_locations = select_locations(allow_all=allow_all)
    if not selected_locations:
        raise ValueError("No locations selected.")

    return selected_locations[0]
