import argparse
import torch

from scipy.spatial.distance import directed_hausdorff


def main():
    """
    Calculates the Directed Hausdorff Distance between two sets of embeddings.

    Usage:
        python script_name.py <embeddings_1> <embeddings_2>

    Arguments:
        embeddings_1: Path to the first embeddings file (PyTorch tensor).
        embeddings_2: Path to the second embeddings file (PyTorch tensor).

    The script loads the embeddings using torch.load and computes the distance
    using SciPy's directed_hausdorff function.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings_1", type=str, metavar="FILE")
    parser.add_argument("embeddings_2", type=str, metavar="FILE")
    args = parser.parse_args()
    embeddings_1 = torch.load(args.embeddings_1)
    embeddings_2 = torch.load(args.embeddings_2)

    print(directed_hausdorff(embeddings_1, embeddings_2))


if __name__ == "__main__":
    main()
