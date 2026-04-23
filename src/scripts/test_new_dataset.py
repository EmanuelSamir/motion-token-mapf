import os
from datasets import load_from_disk
import numpy as np

def test_dataset(path):
    print(f"\n🔍 Testing Dataset at: {path}")
    if not os.path.exists(path):
        print("❌ Error: Path does not exist.")
        return

    try:
        ds = load_from_disk(path)
        print(f"✅ Dataset loaded successfully. Total samples: {len(ds)}")
        
        # Inspect first sample
        sample = ds[0]
        print("\n--- Sample 0 Metadata ---")
        for k, v in sample.items():
            if isinstance(v, list):
                print(f"  {k}: List with len {len(v)}")
            elif hasattr(v, 'shape'):
                print(f"  {k}: Tensor with shape {v.shape}")
            else:
                print(f"  {k}: {v}")

        # Basic integrity checks
        print("\n--- Integrity Checks ---")
        
        # 1. Token check
        tokens = np.array(sample['tokens'])
        valid_range = (tokens >= 0) & (tokens <= 170)
        if valid_range.all():
            print("  ✅ Tokens are within valid range [0, 170]")
        else:
            invalid = tokens[~valid_range]
            print(f"  ❌ Found invalid tokens: {invalid[:5]}")

        # 2. Shape consistency
        history = np.array(sample['history'])
        if len(history.shape) == 3:
            print(f"  ✅ History shape is consistent: {history.shape}")
        else:
            print(f"  ❌ Unexpected history shape: {history.shape}")

        # 3. Agent count check
        num_agents = sample['num_agents']
        if num_agents > 0 and num_agents <= 20:
             print(f"  ✅ num_agents is reasonable: {num_agents}")
        else:
             print(f"  ❌ Odd num_agents value: {num_agents}")

        print("\n🚀 Dataset looks healthy and ready for training!")

    except Exception as e:
        print(f"❌ Failed to test dataset: {e}")

if __name__ == "__main__":
    dataset_path = "data/trajectories/hf_direct_test"
    test_dataset(dataset_path)
