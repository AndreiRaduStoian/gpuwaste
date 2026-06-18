# plot_roofline_scaling.py

import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

data = """
scenario,kernel,scale,predicted_ms,slowdown
compute_scaling,vectoradd,1.0,1.797559,1.00
compute_scaling,vectoradd,0.75,1.797559,1.00
compute_scaling,vectoradd,0.5,1.797559,1.00
compute_scaling,vectoradd,0.25,1.797559,1.00
compute_scaling,compute,1.0,4.960440,1.00
compute_scaling,compute,0.75,6.613920,1.33
compute_scaling,compute,0.5,9.920880,2.00
compute_scaling,compute,0.25,19.841760,4.00
compute_scaling,pointer,1.0,15.129454,1.00
compute_scaling,pointer,0.75,15.129454,1.00
compute_scaling,pointer,0.5,15.129454,1.00
compute_scaling,pointer,0.25,15.129454,1.00
compute_scaling,shared,1.0,0.413232,1.00
compute_scaling,shared,0.75,0.550976,1.33
compute_scaling,shared,0.5,0.826464,2.00
compute_scaling,shared,0.25,1.652928,4.00
compute_scaling,sfu,1.0,0.165293,1.00
compute_scaling,sfu,0.75,0.220390,1.33
compute_scaling,sfu,0.5,0.330586,2.00
compute_scaling,sfu,0.25,0.661171,4.00
memory_scaling,vectoradd,1.0,1.797559,1.00
memory_scaling,vectoradd,0.75,2.396745,1.33
memory_scaling,vectoradd,0.5,3.595118,2.00
memory_scaling,vectoradd,0.25,7.190235,4.00
memory_scaling,compute,1.0,4.960440,1.00
memory_scaling,compute,0.75,4.960440,1.00
memory_scaling,compute,0.5,4.960440,1.00
memory_scaling,compute,0.25,4.960440,1.00
memory_scaling,pointer,1.0,15.129454,1.00
memory_scaling,pointer,0.75,20.172605,1.33
memory_scaling,pointer,0.5,30.258907,2.00
memory_scaling,pointer,0.25,60.517815,4.00
memory_scaling,shared,1.0,0.413232,1.00
memory_scaling,shared,0.75,0.413232,1.00
memory_scaling,shared,0.5,0.599186,1.45
memory_scaling,shared,0.25,1.198373,2.90
memory_scaling,sfu,1.0,0.165293,1.00
memory_scaling,sfu,0.75,0.199729,1.21
memory_scaling,sfu,0.5,0.299593,1.81
memory_scaling,sfu,0.25,0.599186,3.62
combined_scaling,vectoradd,1.0,1.797559,1.00
combined_scaling,vectoradd,0.75,2.396745,1.33
combined_scaling,vectoradd,0.5,3.595118,2.00
combined_scaling,vectoradd,0.25,7.190235,4.00
combined_scaling,compute,1.0,4.960440,1.00
combined_scaling,compute,0.75,6.613920,1.33
combined_scaling,compute,0.5,9.920880,2.00
combined_scaling,compute,0.25,19.841760,4.00
combined_scaling,pointer,1.0,15.129454,1.00
combined_scaling,pointer,0.75,20.172605,1.33
combined_scaling,pointer,0.5,30.258907,2.00
combined_scaling,pointer,0.25,60.517815,4.00
combined_scaling,shared,1.0,0.413232,1.00
combined_scaling,shared,0.75,0.550976,1.33
combined_scaling,shared,0.5,0.826464,2.00
combined_scaling,shared,0.25,1.652928,4.00
combined_scaling,sfu,1.0,0.165293,1.00
combined_scaling,sfu,0.75,0.220390,1.33
combined_scaling,sfu,0.5,0.330586,2.00
combined_scaling,sfu,0.25,0.661171,4.00
"""

df = pd.read_csv(StringIO(data))

for scenario in df["scenario"].unique():
    subset = df[df["scenario"] == scenario]

    plt.figure(figsize=(7, 5))

    for kernel in subset["kernel"].unique():
        kdata = subset[subset["kernel"] == kernel].sort_values("scale")
        plt.plot(kdata["scale"], kdata["slowdown"], marker="o", label=kernel)

    plt.xlabel("Resource scale")
    plt.ylabel("Predicted slowdown")
    plt.title(scenario.replace("_", " ").title())
    plt.grid(True, linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    filename = f"{scenario}.png"
    plt.savefig(filename, dpi=200)
    plt.show()