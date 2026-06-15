#%%
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def scanline_fill(polygon, fill_color='yellow', edge_color='cyan'):
    """
    Implementasi Scan Line Polygon Fill Algorithm
    polygon: list of (x, y) tuples
    """

    # ── 1. Bangun Edge Table ──────────────────────────────────────────
    edges = []
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        # Skip horizontal edges
        if y1 == y2:
            continue

        # Pastikan y_min di bawah
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1

        slope_inv = (x2 - x1) / (y2 - y1)  # dx/dy
        edges.append({
            'y_min': y1,
            'y_max': y2,
            'x': float(x1),          # x saat y = y_min
            'slope_inv': slope_inv
        })

    # ── 2. Tentukan rentang scan line ─────────────────────────────────
    y_min_global = min(e['y_min'] for e in edges)
    y_max_global = max(e['y_max'] for e in edges)

    # ── 3. Setup visualisasi ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax1, ax2 = axes

    # Plot polygon outline di kedua panel
    for ax in axes:
        xs = [p[0] for p in polygon] + [polygon[0][0]]
        ys = [p[1] for p in polygon] + [polygon[0][1]]
        ax.plot(xs, ys, color=edge_color, linewidth=2)
        ax.set_xlim(-2, max(p[0] for p in polygon) + 5)
        ax.set_ylim(-2, max(p[1] for p in polygon) + 5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    ax1.set_title('Polygon (sebelum fill)')
    ax2.set_title('Hasil Scan Line Fill')

    # ── 4. Proses setiap scan line ────────────────────────────────────
    filled_segments = []

    for y in range(int(y_min_global), int(y_max_global)):
        # Active List: edges yang aktif di scan line y ini
        active = [
            e for e in edges
            if e['y_min'] <= y < e['y_max']
        ]

        # Hitung x perpotongan di scan line y
        intersections = []
        for e in active:
            # x = x_start + slope_inv * (y - y_min)
            x_intersect = e['x'] + e['slope_inv'] * (y - e['y_min'])
            intersections.append(x_intersect)

        # Urutkan x dari kiri ke kanan
        intersections.sort()

        # Pasangkan dan isi
        for i in range(0, len(intersections) - 1, 2):
            x_start = intersections[i]
            x_end   = intersections[i + 1]
            filled_segments.append((x_start, x_end, y))

            # Gambar horizontal fill di ax2
            ax2.fill_betweenx(
                [y, y + 1],
                x_start, x_end,
                color=fill_color, alpha=0.7
            )

        # Gambar scan line aktif di ax2 (opsional, tunjukkan progress)
        if len(intersections) >= 2:
            ax2.plot(
                [intersections[0], intersections[-1]],
                [y + 0.5, y + 0.5],
                'r-', linewidth=0.5, alpha=0.4
            )

    # Gambar ulang outline di atas fill
    xs = [p[0] for p in polygon] + [polygon[0][0]]
    ys = [p[1] for p in polygon] + [polygon[0][1]]
    ax2.plot(xs, ys, color=edge_color, linewidth=2)

    plt.tight_layout()
    plt.savefig('scanline_fill_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Total scan lines diproses: {int(y_max_global - y_min_global)}")
    print(f"Total segmen fill: {len(filled_segments)}")


# ── Contoh polygon (mirip ilustrasi di slide) ─────────────────────────
if __name__ == "__main__":
    # Polygon berbentuk rumah sederhana
    polygon_rumah = [
        (10, 0),   # kiri bawah
        (50, 0),   # kanan bawah
        (50, 25),  # kanan tengah
        (40, 25),  # notch kanan
        (40, 30),  # notch atas kanan
        (30, 40),  # puncak kanan
        (20, 30),  # puncak kiri
        (20, 25),  # notch atas kiri
        (10, 25),  # notch kiri
    ]

    print("=== Scan Line Polygon Fill Algorithm ===")
    print(f"Polygon: {len(polygon_rumah)} vertices")
    scanline_fill(polygon_rumah)
# %%
