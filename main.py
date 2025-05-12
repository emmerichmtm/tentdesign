import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull, QhullError
from pathlib import Path

# -----------------------------------------------------------------------------
# Helper: keep only the lowest-z duplicate for every (x, y) pair
# -----------------------------------------------------------------------------

def remove_xy_duplicates_w_lowest_z(points: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Return a point cloud where duplicate (x, y) pairs keep only the lowest z."""
    buckets: dict[tuple[float, float], np.ndarray] = {}
    for x, y, z in points:
        key = (round(x / tol) * tol, round(y / tol) * tol)
        if key not in buckets or z < buckets[key][2]:  # keep the lowest z
            buckets[key] = np.array([x, y, z])
    return np.vstack(list(buckets.values()))

# -----------------------------------------------------------------------------
# Tent class – 100 % DESDEO-free, always saves plots to PNG
# -----------------------------------------------------------------------------

class Tent:
    """3-D convex-hull wrapper that also computes a 2-D floor hull.

    The *plot* method **always** writes a PNG file instead of popping up a GUI
    window. Pass a *save_path* or let it default to *"tent_plot.png"* in the
    current working directory.
    """

    def __init__(self, point_cloud: np.ndarray) -> None:
        self._point_cloud = point_cloud.copy()
        self.main_hull: ConvexHull | None = None
        self.floor_hull: ConvexHull | None = None
        self._is_offset = False
        self.make_hull()

    # ---------------------------------------------------------------------
    # Public properties
    # ---------------------------------------------------------------------
    @property
    def floor_area(self) -> float:
        return self.floor_hull.volume  # 2-D hull uses .volume for polygon area

    @property
    def surface_area(self) -> float:
        return self.main_hull.area - self.floor_area

    @property
    def volume(self) -> float:
        return self.main_hull.volume

    @property
    def min_height(self) -> float:
        z = self._point_cloud[self.main_hull.simplices][:, 2]
        z = z[z > 0]
        return float(np.min(z))

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def make_floor(self) -> None:
        pc = remove_xy_duplicates_w_lowest_z(self._point_cloud)
        self.floor_hull = ConvexHull(pc[:, :2])
        floor_vertices = np.unique(self.floor_hull.simplices)
        self._point_cloud[floor_vertices, 2] = 0.0  # project to z = 0

    def make_hull(self, tries: int = 10) -> None:
        if tries == 0:
            raise RuntimeError("Convex-hull construction failed after offsets.")
        try:
            self.make_floor()
            self.main_hull = ConvexHull(self._point_cloud)
        except QhullError:
            eps = 1e-4
            rng = np.random.default_rng()
            self._point_cloud += rng.uniform(0, eps, self._point_cloud.shape)
            self._is_offset = True
            self.make_hull(tries - 1)

    # ------------------------------------------------------------------
    # Visualisation → PNG
    # ------------------------------------------------------------------
    def plot(self, save_path: str | Path = "tent_plot.png", dpi: int = 300) -> Path:
        """Render the hull and floor; write *save_path* PNG and return the path."""
        save_path = Path(save_path)

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        x, y, z = self._point_cloud.T
        ax.scatter(x, y, z)

        # Main hull wireframe
        for simplex in self.main_hull.simplices:
            simplex = np.append(simplex, simplex[0])
            ax.plot(self._point_cloud[simplex, 0],
                    self._point_cloud[simplex, 1],
                    self._point_cloud[simplex, 2])

        # Floor outline
        for simplex in self.floor_hull.simplices:
            simplex = np.append(simplex, simplex[0])
            ax.plot(self._point_cloud[simplex, 0],
                    self._point_cloud[simplex, 1],
                    self._point_cloud[simplex, 2])

        # Semi-transparent faces
        faces = self._point_cloud[self.main_hull.simplices]
        coll = Poly3DCollection(faces, alpha=0.4)
        ax.add_collection3d(coll)

        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        save_path = save_path.with_suffix(".png")
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path

# -----------------------------------------------------------------------------
# Self-test / demos
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Demo 1 – square-base pyramid
    # ------------------------------------------------------------------
    pts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 1.0],
    ])

    tent = Tent(pts)
    print("=== Demo 1: pyramid ===")
    print(f"Floor area   : {tent.floor_area:.3f}")
    print(f"Surface area : {tent.surface_area:.3f}")
    print(f"Volume       : {tent.volume:.3f}\n")

    out_file = tent.plot("tent_output")  # saves tent_output.png
    print(f"Figure written to → {out_file.resolve()}")

    # ------------------------------------------------------------------
    # Demo 2 – "box3" example from screenshot
    # ------------------------------------------------------------------
    random_points = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],  # floor
        [0.0, 0.0, 0.5],
        [0.0, 1.0, 0.5],
        [1.0, 1.0, 0.5],
        [1.0, 0.0, 0.5],  # floor + 0.5
        [0.0, 0.3, 0.9],
        [0.4, 0.2, 1.1],
        [0.1, 0.4, 1.3],
        [0.4, 0.5, 0.9],
        [0.5, 0.7, 1.3],
        [0.2, 0.3, 0.6],
        [0.2, 0.2, 0.6],
    ])

    box3 = Tent(random_points)
    print("\n=== Demo 2: box3 ===")
    print(f"Floor area   : {box3.floor_area:.3f}")
    print(f"Surface area : {box3.surface_area:.3f}")
    print(f"Volume       : {box3.volume:.3f}\n")

    box3_out = box3.plot("box3_tent")  # saves box3_tent.png
    print(f"Figure written to → {box3_out.resolve()}")