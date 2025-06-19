import argparse

def get_argparse():
    parser = argparse.ArgumentParser(description="General args set up", add_help=False)

    parser.add_argument("--download_path", type=str, default="./data")
    parser.add_argument("--download", type=bool, default=False)
    parser.add_argument("--dataset_name", type=str, default="animals")
    
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--random_seed", type=int, default=10)
    parser.add_argument("--pin_memory", type=bool, default=False, help="If pin memory should be active or not")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers.")
    parser.add_argument("--shuffle", type=bool, default=True, help="Shuffle the dataset.")
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--log_dir", type=str, default="./log_dir")

    parser.add_argument("--isTrainable", type=bool, default=True)
    parser.add_argument("--n_classes", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)



    return parser