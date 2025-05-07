import os
from Preprocess import preprocess

def cls_prepare_data(dataset_dir):
    if dataset_dir is None:
        raise ValueError("No Data")
    
    # based on class labels, go through each image and decide if LCA or RCA
    #RCA: 1 2 3 4 16 16a 16b 16c

    # preprocess all the data

    # Create dataset.yaml file
    yaml_content = f"""
    train: {train_dir}
    val: {val_dir}
    test: {test_dir}
    
    nc: 2
    names: ['LCA', 'RCA']
    """
    
    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Dataset prepared at {dataset_dir}")
    print(f"Training: {len(train_lca)} LCA, {len(train_rca)} RCA")
    print(f"Validation: {len(val_lca)} LCA, {len(val_rca)} RCA")
    print(f"Testing: {len(test_lca)} LCA, {len(test_rca)} RCA")
    
    return yaml_path