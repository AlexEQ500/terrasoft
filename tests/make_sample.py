"""Write a synthetic street scene to sample_site.las (road, kerb, wall, trees, shrub, person, car)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from terrasort.synth import make_scene, write_las

xyz, _ = make_scene()
write_las("sample_site.las", xyz)
print("wrote sample_site.las")
