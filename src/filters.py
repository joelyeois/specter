import torch

def circle2d(N, d):
    """
    Generates 2D array for a filled-in circle.

    Parameters
    ----------
    N : int
        Length of grid in pixels
    d : int
        Diameter in pixels

    Returns
    -------
    circle : tensor, 2D
        2D array with centered circle
    ....
    """
    circle = torch.zeros((N, N))
    x = torch.linspace(-1, 1, N)
    y = x
    [X, Y] = torch.meshgrid(x, y, indexing="ij")
    circle[X**2 + Y**2 <= (d / N)**2] = 1
    return circle