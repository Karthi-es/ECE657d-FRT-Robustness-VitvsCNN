"""Diagnostic script to check data loading"""
from training.data_loaders import create_vggface2_loaders

print("=" * 60)
print("DATA LOADING DIAGNOSTIC")
print("=" * 60)

train_loader, val_loader = create_vggface2_loaders(
    'data/metadata/vggface2_subset.csv',
    batch_size=64,
    num_workers=0,  # No workers for direct diagnosis
)

print(f"\nTrain set:")
print(f"  - Number of batches: {len(train_loader)}")
print(f"  - Total samples: {len(train_loader.dataset)}")
print(f"  - Number of identities: {train_loader.dataset.num_classes}")

print(f"\nVal set:")
print(f"  - Number of batches: {len(val_loader)}")
print(f"  - Total samples: {len(val_loader.dataset)}")

# Load one batch
print(f"\nLoading first batch...")
batch_images, batch_labels = next(iter(train_loader))
print(f"  - Batch image shape: {batch_images.shape}")
print(f"  - Batch labels shape: {batch_labels.shape}")
print(f"  - Image dtype: {batch_images.dtype}")
print(f"  - Image value range: [{batch_images.min():.3f}, {batch_images.max():.3f}]")
print(f"  - Unique labels in batch: {sorted(set(batch_labels.tolist()))}")

# Calculate total training time estimate
num_batches = len(train_loader)
print(f"\nEstimated training time:")
print(f"  - Batches per epoch: {num_batches}")
print(f"  - Epochs: 20")
print(f"  - Total iterations: {num_batches * 20}")
print(f"  - Est. time on RTX3070: ~40-45 minutes (if ~150ms per batch)")

print("\n" + "=" * 60)
