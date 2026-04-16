import pytest
import torch

from .. import nearest_neighbour


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("n,d,nc", [(4096, 3, 32), (83, 8, 17), (1024, 4, 64)])
def test_nearest_neighbour_matches_cdist(n: int, d: int, nc: int) -> None:
    device = torch.device("cuda:0")
    torch.manual_seed(100)
    tensor = torch.randn(n, d, device=device, dtype=torch.float32)
    centroids = torch.randn(nc, d, device=device, dtype=torch.float32)

    expected = torch.cdist(tensor, centroids).argmin(-1)

    for method in ["triton", "torch"]:
        out = nearest_neighbour.nearest_neighbour(tensor, centroids, method=method)
        torch.testing.assert_close(out, expected)

        out.fill_(-1)
        nearest_neighbour.nearest_neighbour(tensor, centroids, out=out, method=method)
        torch.testing.assert_close(out, expected)
