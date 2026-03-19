"""Check which identity directories have images"""
from pathlib import Path

test_root = Path("data/vggface2/test")
identity_dirs = sorted([d for d in test_root.iterdir() if d.is_dir()])

with_images = []
without_images = []

for ident_dir in identity_dirs:
    images = list(ident_dir.glob("*.jpg")) + list(ident_dir.glob("*.jpeg")) + list(ident_dir.glob("*.png"))
    if images:
        with_images.append(ident_dir.name)
    else:
        without_images.append(ident_dir.name)

print(f"Identity directories WITH images: {len(with_images)}")
print(f"Identity directories WITHOUT images: {len(without_images)}")
print(f"Total identity directories: {len(identity_dirs)}")
print(f"\nCompletion rate: {len(with_images)}/{len(identity_dirs)} = {100*len(with_images)/len(identity_dirs):.1f}%")

if len(with_images) <= 30:
    print(f"\nIdentities WITH images:")
    for name in sorted(with_images):
        print(f"  {name}")
