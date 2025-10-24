#!/usr/bin/env python3
"""
Render NetworkX graph with US state boundaries to SVG.

Takes a pickled graph file and renders state polygons as white shapes
with dark grey borders on a transparent background.
"""

import csv
import dataclasses
import math
import pickle
import sys
import typing

import matplotlib.axes
import matplotlib.collections
import matplotlib.patches
import matplotlib.pyplot
import numpy
import shapely.wkt

# Grid aspect ratio overrides: (width_factor, height_factor)
# Values < 1.0 make narrower/shorter, > 1.0 make wider/taller
GRID_ASPECT_OVERRIDES = {
    "CA": (0.8, 1.3),  # More vertical (narrower, taller)
    "IN": (0.8, 1.3),  # More vertical
    "IL": (0.8, 1.3),  # More vertical
    "KY": (1.4, 0.7),  # More horizontal (wider, shorter)
    "TN": (1.4, 0.7),  # More horizontal
    "NC": (1.4, 0.7),  # More horizontal
}


def parse_wkt_polygon(wkt):
    """Parse WKT polygon string and return matplotlib matplotlib.patches."""
    try:
        geom = shapely.wkt.loads(wkt)
        polygons = []

        if geom.geom_type == "Polygon":
            polygons = [geom]
        elif geom.geom_type == "MultiPolygon":
            polygons = list(geom.geoms)

        patches_list = []
        for poly in polygons:
            # Get exterior coordinates and round to nearest whole number
            exterior_coords = [(round(x), round(y)) for x, y in poly.exterior.coords]
            if exterior_coords:
                patch = matplotlib.patches.Polygon(exterior_coords, closed=True)
                patches_list.append(patch)

            # Handle holes (interior rings) - round coordinates
            for interior in poly.interiors:
                hole_coords = [(round(x), round(y)) for x, y in interior.coords]
                if hole_coords:
                    # Note: Proper hole handling would require Path objects,
                    # but for simplicity we'll let matplotlib handle this automatically
                    pass

        return patches_list
    except Exception as e:
        print(f"Error parsing WKT: {e}")
        return []


@dataclasses.dataclass
class StateLabel:
    """Represents a state label with position, text, and rendering properties."""

    state: str
    seats: int
    orig_x: float
    orig_y: float
    x: float = None
    y: float = None
    width: float = None
    height: float = None

    # Internal implementation detail
    _fontsize: float = 7

    def __post_init__(self):
        """Initialize position and calculate dimensions after creation."""
        if self.x is None:
            self.x = self.orig_x
        if self.y is None:
            self.y = self.orig_y
        self._calculate_dimensions()

    @property
    def text(self) -> str:
        """Generate the display text for this label."""
        return self.state

    def _calculate_dimensions(self):
        """Calculate label dimensions based on text, font size, and seat dots grid."""
        # More generous approximation for bold monospace text
        char_width = (
            self._fontsize * 3.0
        )  # pixels per character (increased by 50% more to ensure proper text coverage)
        char_height = self._fontsize * 1.0  # line height closer to actual font size

        lines = self.text.split("\n")
        max_line_length = max(len(line) for line in lines)
        num_lines = len(lines)

        text_width = max_line_length * char_width
        text_height = char_height * num_lines

        # Calculate seat dots grid dimensions
        grid_cols, grid_rows = self._calculate_grid_dimensions(self.seats)
        dot_size = 16  # 16px rectangles (doubled from 8px)
        dot_margin = 4  # 4px margins (doubled from 2px)

        if grid_cols > 0 and grid_rows > 0:
            grid_width = grid_cols * dot_size + (grid_cols - 1) * dot_margin
            grid_height = grid_rows * dot_size + (grid_rows - 1) * dot_margin
        else:
            grid_width = 0
            grid_height = 0

        # Add spacing between text and grid
        padding = (
            44 if grid_height > 0 else 0
        )  # Maximum padding to completely eliminate overlap

        # Layout: text on top, grid below, left-aligned
        self.width = max(text_width, grid_width)
        self.height = text_height + padding + grid_height

        # Debug output for dimension validation
        if grid_cols > 0 and grid_rows > 0:
            print(
                f"DEBUG {self.state}: text='{self.text}' {text_width:.0f}x{text_height:.0f}, "
                f"grid={grid_width:.0f}x{grid_height:.0f} ({grid_cols}x{grid_rows} for {self.seats} seats), "
                f"padding={padding:.0f}, total={self.width:.0f}x{self.height:.0f}"
            )
        else:
            print(
                f"DEBUG {self.state}: text='{self.text}' {text_width:.0f}x{text_height:.0f}, "
                f"no grid, total={self.width:.0f}x{self.height:.0f}"
            )

    def _calculate_grid_dimensions(self, seats: int) -> tuple[int, int]:
        """Calculate the grid dimensions (cols, rows) for the given number of seats."""
        if seats <= 0:
            return (0, 0)

        # Start with square grid as baseline
        base_size = math.ceil(math.sqrt(seats))

        # Apply aspect ratio overrides if this state has them
        if self.state in GRID_ASPECT_OVERRIDES:
            width_factor, height_factor = GRID_ASPECT_OVERRIDES[self.state]

            # Adjust grid dimensions based on aspect ratio
            cols = max(1, round(base_size * width_factor))
            rows = math.ceil(seats / cols)

            return (cols, rows)
        else:
            # Default square behavior - calculate actual rows needed
            rows = math.ceil(seats / base_size)
            return (base_size, rows)

    def get_map_dimensions(self, map_width: float) -> tuple[float, float]:
        """Calculate dimensions scaled to map coordinates."""
        char_scale = map_width / 200  # Scale factor for text size vs map

        # Use the calculated bbox dimensions and scale to map coordinates
        data_width = (
            self.width * char_scale / 10
        )  # Scale relative to original 10pt font
        data_height = (
            self.height * char_scale / 10
        )  # Scale relative to original 10pt font

        # Add padding to bbox for better visual separation
        char_scale = map_width / 200
        margin_4px = 4 * char_scale / 10  # Convert 4px to map coordinates

        data_width *= 1.05  # 5% padding
        data_height *= 1.0  # No extra padding on height

        # Add 4px margin on all sides
        data_width += 2 * margin_4px  # 4px on each side
        data_height += 2 * margin_4px  # 4px on each side

        # Debug output for scaling
        print(
            f"DEBUG {self.state} scaling: raw={self.width:.0f}x{self.height:.0f}, "
            f"scaled={data_width / 10:.0f}x{data_height / 10:.0f}, "
            f"final={data_width:.0f}x{data_height:.0f}"
        )

        return data_width, data_height

    def render_bbox(self, ax: matplotlib.axes.Axes, map_width: float) -> None:
        """Render the white bounding box for this label."""
        data_width, data_height = self.get_map_dimensions(map_width)

        # Round all coordinates for cleaner SVG output
        label_x = round(self.x)
        label_y = round(self.y)
        bbox_width = round(data_width)
        bbox_height = round(data_height)

        # Position bbox centered on label position
        # This creates the coordinate system that text and grid use
        bbox_left = label_x - bbox_width // 2
        bbox_bottom = label_y - bbox_height // 2

        bbox_rect = matplotlib.patches.Rectangle(
            (bbox_left, bbox_bottom),
            bbox_width,
            bbox_height,
            linewidth=0,
            edgecolor="none",
            facecolor="white",
            alpha=0.75,
        )
        ax.add_patch(bbox_rect)

    def render_text(self, ax: matplotlib.axes.Axes, map_width: float) -> None:
        """Render the text label."""
        # Round coordinates for cleaner SVG output
        label_x = round(self.x)
        label_y = round(self.y)

        # Get actual dimensions for positioning using correct map_width
        data_width, data_height = self.get_map_dimensions(map_width)

        # Calculate 4px margin in map coordinates
        char_scale = map_width / 200
        margin_4px = 4 * char_scale / 10

        # Position text at top-left of the content area (inside margin)
        # Label center is at (label_x, label_y), so bbox goes from:
        # left: label_x - data_width/2  to  right: label_x + data_width/2
        # bottom: label_y - data_height/2  to  top: label_y + data_height/2
        text_x = label_x - data_width // 2 + margin_4px  # Left edge + margin
        text_y = label_y + data_height // 2 - margin_4px  # Top edge - margin

        ax.text(
            text_x,
            text_y,
            self.text,
            ha="left",
            va="top",
            fontsize=self._fontsize,
            fontweight="bold",
            color="black",
            fontfamily="monospace",
        )

    def render_seat_dots(
        self, ax: matplotlib.axes.Axes, map_width: float, efficiency_gap: float
    ) -> None:
        """Render the seat dots grid with efficiency gap colors."""
        if self.seats <= 0:
            return

        # Grid parameters
        grid_cols, grid_rows = self._calculate_grid_dimensions(self.seats)
        dot_size = 16  # pixels (doubled from 8px)
        dot_margin = 4  # pixels (doubled from 2px)

        # Scale to map coordinates
        char_scale = map_width / 200
        scaled_dot_size = dot_size * char_scale / 10
        scaled_dot_margin = dot_margin * char_scale / 10

        # Get label position and dimensions
        label_x = round(self.x)
        label_y = round(self.y)
        data_width, data_height = self.get_map_dimensions(map_width)

        # Calculate grid positioning - left-aligned, below text with maximum spacing
        text_height = self._fontsize * 1.0 * char_scale / 10
        padding = 44 * char_scale / 10

        # Calculate 4px margin in map coordinates
        margin_4px = 4 * char_scale / 10

        # Grid starts at left edge of content area (inside margin), positioned below text + padding
        grid_start_x = label_x - data_width // 2 + margin_4px  # Left edge + margin
        grid_start_y = (
            label_y + data_height // 2 - margin_4px - text_height - padding
        )  # Below text + padding, inside margin

        # Determine color based on efficiency gap
        dot_color = self._efficiency_gap_to_color(efficiency_gap)

        # Draw seats as filled rectangles
        seats_drawn = 0
        for row in range(grid_rows):
            for col in range(grid_cols):
                if seats_drawn >= self.seats:
                    break

                dot_x = grid_start_x + col * (scaled_dot_size + scaled_dot_margin)
                dot_y = grid_start_y - row * (scaled_dot_size + scaled_dot_margin)

                dot_rect = matplotlib.patches.Rectangle(
                    (dot_x, dot_y),
                    scaled_dot_size,
                    scaled_dot_size,
                    linewidth=0,
                    edgecolor="none",
                    facecolor=dot_color,
                    alpha=1.0,
                )
                ax.add_patch(dot_rect)
                seats_drawn += 1

            if seats_drawn >= self.seats:
                break

    def _efficiency_gap_to_color(self, efficiency_gap: float) -> str:
        """
        Convert efficiency gap to color.

        Args:
            efficiency_gap: Efficiency gap as fraction (-0.2 to +0.2 range)
                           Positive = pro-Republican advantage
                           Negative = pro-Democratic advantage

        Returns:
            Hex color string
        """
        # Clamp to -20% to +20% range
        clamped_gap = max(-0.2, min(0.2, efficiency_gap))

        if abs(clamped_gap) < 0.01:  # Very close to zero, use light gray
            return "#D3D3D3"

        if clamped_gap > 0:
            # Pro-Republican (positive gap), red gradient to rgb(199, 28, 54)
            intensity = clamped_gap / 0.2  # 0 to 1
            # Interpolate from light gray (211, 211, 211) to red (199, 28, 54)
            red = int(211 + (199 - 211) * intensity)
            green = int(211 + (28 - 211) * intensity)
            blue = int(211 + (54 - 211) * intensity)
            return f"#{red:02x}{green:02x}{blue:02x}"
        else:
            # Pro-Democratic (negative gap), blue gradient to rgb(0, 73, 168)
            intensity = abs(clamped_gap) / 0.2  # 0 to 1
            # Interpolate from light gray (211, 211, 211) to blue (0, 73, 168)
            red = int(211 + (0 - 211) * intensity)
            green = int(211 + (73 - 211) * intensity)
            blue = int(211 + (168 - 211) * intensity)
            return f"#{red:02x}{green:02x}{blue:02x}"

    def to_force_dict(self, map_width: float) -> dict[str, typing.Any]:
        """Convert to dictionary format for force-directed algorithm."""
        data_width, data_height = self.get_map_dimensions(map_width)
        return {
            "text": self.text,
            "orig_x": self.orig_x,
            "orig_y": self.orig_y,
            "x": self.x,
            "y": self.y,
            "width": data_width,
            "height": data_height,
        }


def calculate_label_bbox(text, fontsize=10):
    """Estimate bounding box dimensions for a text label (supports multiline)."""
    # More accurate approximation for bold text
    char_width = fontsize * 0.8  # pixels per character (bold text is wider)
    char_height = fontsize * 1.4  # line height with padding

    lines = text.split("\n")
    max_line_length = max(len(line) for line in lines)
    num_lines = len(lines)

    width = max_line_length * char_width
    height = char_height * num_lines

    return width, height


def bboxes_overlap(label1, label2):
    """Check if two label bounding boxes overlap."""
    # Get bbox corners
    left1 = label1["x"] - label1["width"] / 2
    right1 = label1["x"] + label1["width"] / 2
    top1 = label1["y"] + label1["height"] / 2
    bottom1 = label1["y"] - label1["height"] / 2

    left2 = label2["x"] - label2["width"] / 2
    right2 = label2["x"] + label2["width"] / 2
    top2 = label2["y"] + label2["height"] / 2
    bottom2 = label2["y"] - label2["height"] / 2

    # Check if they don't overlap (easier to check)
    if right1 <= left2 or right2 <= left1 or top1 <= bottom2 or top2 <= bottom1:
        return False
    return True


def has_any_overlaps(labels):
    """Check if any labels have overlapping bounding boxes."""
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if bboxes_overlap(labels[i], labels[j]):
                return True
    return False


def force_directed_label_layout(
    labels, max_iterations=10000, k_attract=0.005, k_repel=50000
):
    """
    Apply force-directed algorithm to separate overlapping labels.
    Stops when no overlaps are detected or max iterations reached.

    Args:
        labels: list of dicts with 'text', 'orig_x', 'orig_y', 'x', 'y', 'width', 'height'
        max_iterations: Maximum number of iterations
        k_attract: Attraction force constant (toward original position)
        k_repel: Repulsion force constant (away from overlapping labels)
    """
    positions = numpy.array([[label["x"], label["y"]] for label in labels])
    # orig_positions = numpy.array([[label['orig_x'], label['orig_y']] for label in labels])  # Unused while attraction disabled

    for iteration in range(max_iterations):
        forces = numpy.zeros_like(positions)

        # Temporarily disable attraction force to debug repulsion
        # attract_force = k_attract * (orig_positions - positions)
        # forces += attract_force

        # Update label positions for accurate overlap checking
        for i, label in enumerate(labels):
            label["x"], label["y"] = positions[i]

        # Count overlaps for debugging
        overlaps_found = False
        overlap_count = 0

        # Repulsion forces between overlapping labels
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                label_i = labels[i]
                label_j = labels[j]

                # Check if bounding boxes overlap
                if bboxes_overlap(label_i, label_j):
                    overlaps_found = True
                    overlap_count += 1

                    # Calculate repulsion force
                    dx = positions[j][0] - positions[i][0]
                    dy = positions[j][1] - positions[i][1]

                    # Avoid division by zero
                    if abs(dx) < 1:
                        dx = 1 if dx >= 0 else -1
                    if abs(dy) < 1:
                        dy = 1 if dy >= 0 else -1

                    distance = max(numpy.sqrt(dx**2 + dy**2), 1)
                    unit_dx = dx / distance
                    unit_dy = dy / distance

                    # Stronger repulsion force that scales with overlap
                    overlap_x = max(
                        0, (label_i["width"] + label_j["width"]) / 2 - abs(dx)
                    )
                    overlap_y = max(
                        0, (label_i["height"] + label_j["height"]) / 2 - abs(dy)
                    )
                    overlap_severity = (overlap_x * overlap_y) / max(
                        label_i["width"] * label_i["height"], 1
                    )

                    repel_strength = (
                        k_repel * (1 + overlap_severity * 10) / max(distance, 50)
                    )

                    # Apply repulsive forces (push apart)
                    forces[i][0] -= repel_strength * unit_dx
                    forces[i][1] -= repel_strength * unit_dy
                    forces[j][0] += repel_strength * unit_dx
                    forces[j][1] += repel_strength * unit_dy

        print(f"Iteration {iteration + 1}: {overlap_count} overlaps found")

        # Debug: check if positions are actually changing
        old_positions = positions.copy()

        # Apply forces with much larger movement to match coordinate system scale
        damping = 1.0
        movement_scale = 1000  # Scale movement to match map coordinate system
        positions += forces * damping * movement_scale

        # Check how much things moved
        movement = numpy.sqrt(numpy.sum((positions - old_positions) ** 2, axis=1))
        max_movement = numpy.max(movement)
        avg_movement = numpy.mean(movement)
        print(f"  Max movement: {max_movement:.1f}, Avg movement: {avg_movement:.1f}")

        # Also check force magnitudes
        force_magnitudes = numpy.sqrt(numpy.sum(forces**2, axis=1))
        max_force = numpy.max(force_magnitudes)
        avg_force = numpy.mean(force_magnitudes)
        print(f"  Max force: {max_force:.1f}, Avg force: {avg_force:.1f}")

        # Check if we've resolved all overlaps
        if not overlaps_found:
            print(f"No overlaps detected after {iteration + 1} iterations")
            break

    print(f"Final check: {overlap_count} overlaps remaining")

    # Update label positions - no distance constraints
    for i, label in enumerate(labels):
        label["x"] = positions[i][0]
        label["y"] = positions[i][1]


def calculate_efficiency_gap(district_votes):
    """
    Calculate the efficiency gap for a state's districts.

    Args:
        district_votes: List of tuples (votes_dem, votes_rep) for each district

    Returns:
        Efficiency gap as a fraction (positive = pro-Republican advantage, negative = pro-Democratic advantage)
    """
    total_votes = 0
    dem_wasted = 0
    rep_wasted = 0

    for votes_dem, votes_rep in district_votes:
        district_total = votes_dem + votes_rep
        if district_total == 0:
            continue

        total_votes += district_total

        # Calculate wasted votes
        if votes_dem > votes_rep:
            # Democrat wins
            # Republican wasted votes = all their votes (they lost)
            rep_wasted += votes_rep
            # Democrat wasted votes = votes above 50% + 1 threshold
            dem_wasted += votes_dem - (district_total // 2 + 1)
        else:
            # Republican wins
            # Democrat wasted votes = all their votes (they lost)
            dem_wasted += votes_dem
            # Republican wasted votes = votes above 50% + 1 threshold
            rep_wasted += votes_rep - (district_total // 2 + 1)

    if total_votes == 0:
        return 0.0

    # Efficiency gap = (Dem wasted - Rep wasted) / Total votes
    # Positive means Republicans have advantage (more Dem wasted votes)
    # Negative means Democrats have advantage (more Rep wasted votes)
    return (dem_wasted - rep_wasted) / total_votes


def load_2024_vote_data(tsv_file):
    """Load 2024 vote data from TSV file and calculate efficiency gaps."""
    state_districts = {}  # state -> list of (votes_dem, votes_rep) tuples
    state_totals = {}  # state -> {votes_dem_est, votes_rep_est, seats}

    with open(tsv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            if row["year"] == "2024":
                state = row["stateabrev"]
                votes_dem_est = (
                    int(row["votes_dem_est"].replace(",", ""))
                    if row["votes_dem_est"]
                    else 0
                )
                votes_rep_est = (
                    int(row["votes_rep_est"].replace(",", ""))
                    if row["votes_rep_est"]
                    else 0
                )

                # Store district-level data for efficiency gap calculation
                if state not in state_districts:
                    state_districts[state] = []
                state_districts[state].append((votes_dem_est, votes_rep_est))

                # Store state totals
                if state not in state_totals:
                    state_totals[state] = {
                        "votes_dem_est": 0,
                        "votes_rep_est": 0,
                        "seats": 0,
                        "efficiency_gap": 0.0,
                    }

                state_totals[state]["votes_dem_est"] += votes_dem_est
                state_totals[state]["votes_rep_est"] += votes_rep_est
                state_totals[state]["seats"] += 1

    # Calculate efficiency gaps for each state
    for state in state_totals:
        if state in state_districts:
            efficiency_gap = calculate_efficiency_gap(state_districts[state])
            state_totals[state]["efficiency_gap"] = efficiency_gap
            print(
                f"DEBUG {state}: Efficiency gap = {efficiency_gap:.3f} ({efficiency_gap * 100:.1f}%)"
            )

    return state_totals


def render_graph(graph_file, output_file):
    """Render the pickled graph to SVG."""
    # Load the graph
    print(f"Loading graph from {graph_file}...")
    with open(graph_file, "rb") as f:
        G = pickle.load(f)

    print(f"Loaded graph with {G.number_of_nodes()} nodes")

    # Load 2024 vote data
    print("Loading 2024 vote data...")
    vote_data = load_2024_vote_data(
        "PlanScore Production Data (2025) - USH Outcomes (2025).tsv"
    )
    print(f"Loaded vote data for {len(vote_data)} states")

    # Set up the plot with specified dimensions (962px width)
    # Calculate height to maintain aspect ratio while ensuring 962px width
    fig, ax = matplotlib.pyplot.subplots(figsize=(9.62, 4.0))
    ax.set_aspect("equal")

    # Remove axes and make background transparent
    ax.set_axis_off()
    fig.patch.set_alpha(0.0)

    all_patches = []

    # Process each state polygon
    for state_code, data in G.nodes(data=True):
        wkt = data.get("wkt")
        if wkt:
            state_patches = parse_wkt_polygon(wkt)
            all_patches.extend(state_patches)

    print(f"Created {len(all_patches)} polygon patches")

    # Create patch collection
    if all_patches:
        collection = matplotlib.collections.PatchCollection(
            all_patches,
            facecolor="white",
            edgecolor="#333333",  # dark grey
            linewidth=1.0,
            alpha=1.0,
        )
        ax.add_collection(collection)

        # Set plot limits based on patch extents
        ax.autoscale()

    # Get map bounds for scaling calculations
    map_bounds = ax.get_xlim(), ax.get_ylim()
    map_width = map_bounds[0][1] - map_bounds[0][0]

    # Prepare labels for force-directed positioning using 2024 data
    state_labels: list[StateLabel] = []
    vote_data_with_labels = {}  # Store vote data for rendering

    for state_code, data in G.nodes(data=True):
        x = data.get("x")
        y = data.get("y")

        # Use 2024 data if available, otherwise fall back to graph data
        if state_code in vote_data:
            seats = vote_data[state_code]["seats"]
            efficiency_gap = vote_data[state_code]["efficiency_gap"]
        else:
            seats = data.get("seats", 0)
            efficiency_gap = 0.0

        if x is not None and y is not None and seats > 0:
            state_label = StateLabel(
                state=state_code,
                seats=seats,
                orig_x=x,
                orig_y=y,
            )
            state_labels.append(state_label)
            vote_data_with_labels[state_code] = {
                "efficiency_gap": efficiency_gap,
            }

    print(f"Applying force-directed layout to {len(state_labels)} labels...")

    # Debug: print some label dimensions
    for i, state_label in enumerate(state_labels[:5]):  # Print first 5
        data_width, data_height = state_label.get_map_dimensions(map_width)
        print(
            f"Label {state_label.text}: width={data_width:.0f}, height={data_height:.0f}"
        )

    # Convert StateLabels to force dict format for the algorithm
    force_labels: list[dict[str, typing.Any]] = [
        label.to_force_dict(map_width) for label in state_labels
    ]

    # Apply force-directed algorithm to separate overlapping labels
    force_directed_label_layout(force_labels)

    # Update StateLabel positions from force results
    for state_label, force_label in zip(state_labels, force_labels):
        state_label.x = force_label["x"]
        state_label.y = force_label["y"]

    # Draw labels at their adjusted positions
    for state_label in state_labels:
        # Get efficiency gap data for this state
        state_data = vote_data_with_labels.get(
            state_label.state, {"efficiency_gap": 0.0}
        )

        # Render bounding box, text, and seat dots using StateLabel methods
        state_label.render_bbox(ax, map_width)
        state_label.render_text(ax, map_width)
        state_label.render_seat_dots(
            ax,
            map_width,
            state_data["efficiency_gap"],
        )

    # Reset plot limits to original bounds (before labels were added) with rounded bounds
    if all_patches:
        ax.autoscale()
        # Get current limits and round them to integers for cleaner SVG
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(round(xlim[0]), round(xlim[1]))
        ax.set_ylim(round(ylim[0]), round(ylim[1]))

    # Save as SVG with specific DPI to achieve 962px width
    # figsize 9.62 inches * 100 DPI = 962 pixels width
    print(f"Saving to {output_file}...")
    matplotlib.pyplot.savefig(
        output_file,
        format="svg",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
        dpi=100,
    )
    matplotlib.pyplot.close()

    # Post-process SVG to set exact width of 962px
    print("Post-processing SVG to set width to 962px...")
    import re

    with open(output_file, "r") as f:
        svg_content = f.read()

    # Extract current width and height in points
    width_match = re.search(r'width="([0-9.]+)pt"', svg_content)
    height_match = re.search(r'height="([0-9.]+)pt"', svg_content)

    if width_match and height_match:
        current_width_pt = float(width_match.group(1))
        current_height_pt = float(height_match.group(1))

        # Calculate height to maintain aspect ratio with 962px width
        aspect_ratio = current_height_pt / current_width_pt
        new_height_px = int(962 * aspect_ratio)

        # Replace width and height with pixel values
        svg_content = re.sub(r'width="[0-9.]+pt"', 'width="962px"', svg_content)
        svg_content = re.sub(
            r'height="[0-9.]+pt"', f'height="{new_height_px}px"', svg_content
        )

        with open(output_file, "w") as f:
            f.write(svg_content)

        print(f"SVG dimensions updated to 962x{new_height_px} pixels")
    else:
        print("Could not find width/height in SVG to modify")

    print(f"SVG saved successfully to {output_file}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python render-graph.py <graph.pickle> <output.svg>")
        sys.exit(1)

    graph_file = sys.argv[1]
    output_file = sys.argv[2]

    render_graph(graph_file, output_file)


if __name__ == "__main__":
    main()
