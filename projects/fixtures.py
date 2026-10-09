"""Synthetic software fixtures, not real-world evaluation data."""
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

def generate(destination):
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    images = []
    for i in range(40):
        label = "healthy" if i%2==0 else "leaf_spot"
        leaf = rng.integers(0,30,(64,64,3),dtype=np.uint8)
        leaf[:,:,1] += 110
        if label == "leaf_spot":
            for _ in range(12):
                x,y = rng.integers(5,59,2)
                cv2.circle(leaf,(int(x),int(y)),4,(20,45,120),-1)
        filename = f"leaf_{i}.png"
        cv2.imwrite(str(root / filename),leaf)
        images.append(dict(path=filename,crop="tomato" if i%4<2 else "potato",humidity=float(rng.uniform(40,90)),temperature=float(rng.uniform(18,35)),label=label))
    pd.DataFrame(images).to_csv(root / "agri.csv",index=False)
    (root / "NOTICE.txt").write_text("SYNTHETIC SOFTWARE TEST FIXTURES. Not real-world evaluation data.", encoding="utf-8")
    return root

if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination")
    print(generate(parser.parse_args().destination))
