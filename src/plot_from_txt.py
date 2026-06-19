import re
import matplotlib.pyplot as plt

FERMI_TEXT = """gpu       wg_size   w/g groups   occ       cycles        WPC       norm
-----------------------------------------------------------------------
Fermi          32     1      1     1     14808.00    0.00007     100.00
Fermi          32     1      2     2     14810.00    0.00014     199.97
Fermi          64     2      1     2     15319.00    0.00013     193.33
Fermi          32     1      4     4     14814.00    0.00027     399.84
Fermi          64     2      2     4     15323.00    0.00026     386.56
Fermi         128     4      1     4     16341.00    0.00024     362.47
Fermi          32     1      8     8     14822.00    0.00054     799.24
Fermi          64     2      4     8     15331.00    0.00052     772.71
Fermi         128     4      2     8     16349.00    0.00049     724.59
Fermi         256     8      1     8     18385.00    0.00044     644.35
Fermi          32     1     16    16     14839.00    0.00108    1596.66
Fermi          64     2      8    16     15604.00    0.00103    1518.38
Fermi         128     4      4    16     17635.00    0.00091    1343.51
Fermi         256     8      2    16     19925.00    0.00080    1189.10
Fermi         512    16      1    16     22473.00    0.00071    1054.28
Fermi          32     1     32    32     29871.00    0.00107    1586.34
Fermi          64     2     16    32     20480.00    0.00156    2313.75
Fermi         128     4      8    32     21764.00    0.00147    2177.25
Fermi         256     8      4    32     23534.00    0.00136    2013.50
Fermi         512    16      2    32     26329.00    0.00122    1799.75
Fermi        1024    32      1    32     34219.00    0.00094    1384.77
Fermi          32     1     48    48     29921.00    0.00160    2375.54
Fermi          64     2     24    48     30696.00    0.00156    2315.56
Fermi         128     4     12    48     32993.00    0.00145    2154.35
Fermi         256     8      6    48     34896.00    0.00138    2036.86
Fermi         512    16      3    48     35534.00    0.00135    2000.29"""

PASCAL_TEXT = """gpu       wg_size   w/g groups   occ       cycles        WPC       norm
-----------------------------------------------------------------------
Pascal         32     1      1     1     19386.00    0.00005     100.00
Pascal         32     1      2     2     19388.25    0.00010     199.98
Pascal         64     2      1     2     19960.00    0.00010     194.25
Pascal         32     1      4     4     19392.75    0.00021     399.86
Pascal         64     2      2     4     19964.50    0.00020     388.41
Pascal        128     4      1     4     21108.00    0.00019     367.37
Pascal         32     1      8     8     19401.75    0.00041     799.35
Pascal         64     2      4     8     19973.50    0.00040     776.47
Pascal        128     4      2     8     21117.00    0.00038     734.42
Pascal        256     8      1     8     23404.00    0.00034     662.66
Pascal         32     1     16    16     19419.75    0.00082    1597.22
Pascal         64     2      8    16     19991.50    0.00080    1551.54
Pascal        128     4      4    16     21135.00    0.00076    1467.59
Pascal        256     8      2    16     23485.00    0.00068    1320.74
Pascal        512    16      1    16     27996.00    0.00057    1107.93
Pascal         32     1     32    32     19455.75    0.00164    3188.53
Pascal         64     2     16    32     20027.50    0.00160    3097.50
Pascal        128     4      8    32     21171.00    0.00151    2930.20
Pascal        256     8      4    32     23521.00    0.00136    2637.44
Pascal        512    16      2    32     28095.00    0.00114    2208.05
Pascal       1024    32      1    32     37243.50    0.00086    1665.67
Pascal         32     1     64    64     38959.00    0.00164    3184.64
Pascal         64     2     32    64     40111.25    0.00160    3093.16
Pascal        128     4     16    64     43529.50    0.00147    2850.26
Pascal        256     8      8    64     47580.00    0.00135    2607.62
Pascal        512    16      4    64     36803.75    0.00174    3371.13
Pascal       1024    32      2    64     37315.75    0.00172    3324.88
"""

def parse_rows(text):
    rows = []
    pattern = re.compile(
        r"^(Fermi|Pascal)\s+"
        r"(?P<wg_size>\d+)\s+"
        r"(?P<warps_per_group>\d+)\s+"
        r"(?P<groups>\d+)\s+"
        r"(?P<occupancy>\d+)\s+"
        r"(?P<cycles>[0-9.]+)"
    )

    for line in text.splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue

        d = m.groupdict()
        rows.append({
            "gpu": m.group(1),
            "work_group_size": int(d["wg_size"]),
            "warps_per_group": int(d["warps_per_group"]),
            "groups": int(d["groups"]),
            "occupancy": int(d["occupancy"]),
            "cycles": float(d["cycles"]),
        })

    return rows

def add_ipc_alu(rows, compute_instr_per_warp=256):
    for r in rows:
        r["ipc_alu"] = r["occupancy"] * compute_instr_per_warp / r["cycles"]
    return rows

fermi_rows = add_ipc_alu(parse_rows(FERMI_TEXT))
pascal_rows = add_ipc_alu(parse_rows(PASCAL_TEXT))

def plot_sorted(rows, title, path):
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: (r["occupancy"], r["work_group_size"]))

    xs = list(range(len(rows)))
    ys = [r["ipc_alu"] for r in rows]

    plt.figure(figsize=(10, 4))
    plt.plot(xs, ys, marker="o", linewidth=1)

    plt.title(title)
    plt.xlabel("Configurations sorted by occupancy, then work group size")
    plt.ylabel("ALU IPC")

    plt.grid(axis="y", alpha=0.3)
    plt.xticks([])

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.show()

plot_sorted(fermi_rows, "Fermi iterative barrier", "../results/pipeline_results/fermi_iter_barrier.png")
plot_sorted(pascal_rows, "Pascal iterative barrier", "../results/pipeline_results/pascal_iter_barrier.png")