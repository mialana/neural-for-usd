from imports import *


class Renderer:
    def __init__(self):
        self.n_training = 100
        self.near, self.far = 2., 6.
        testimg_idx = 101

        if not os.path.exists("tiny_nerf_data.npz"):
            wget.download(
                "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz"
            )

        npz_file: NpzFile = np.load("tiny_nerf_data.npz")

        # Gather as torch tensors
        self.images = torch.from_numpy(npz_file["images"][:self.n_training]).to(device)
        self.poses = torch.from_numpy(npz_file["poses"]).to(device)
        self.focal = (
            torch.from_numpy(npz_file["focal"]).to(dtype=torch.float32).to(device)
        )
        self.testimg = torch.from_numpy(npz_file["images"][testimg_idx]).to(device)
        self.testpose = torch.from_numpy(npz_file["poses"][testimg_idx]).to(device)

        # Grab rays from sample image
        self.height, self.width = self.images.shape[1:3]

    def get_rays(
        self,
        height: Optional[int],
        width: Optional[int],
        focal_length: Optional[float],
        c2w: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Find origin and direction of rays through every pixel and camera origin.
        """

        if height is None:
            height = self.height
        if width is None:
            width = self.width
        if focal_length is None:
            focal_length = self.focal
        if c2w is None:
            c2w = self.testpose

        # Apply pinhole camera model to gather directions at each pixel
        # i is a [width, height] grid with x-values of pixel at coordinate
        # j is a [width, height] grid with x-values of pixel at coordinate
        i, j = torch.meshgrid(
            torch.arange(width, dtype=torch.float32).to(c2w),
            torch.arange(height, dtype=torch.float32).to(c2w),
            indexing="ij",
        )
        logging.debug("i Shape: ", i.shape)
        logging.debug("j Shape: ", j.shape)

        # swaps the last two dimensions (should be just 2, to get i and j shaped [height, width].
        i, j = i.transpose(-1, -2), j.transpose(-1, -2)
        logging.debug("i Shape Transposed: ", i.shape)
        logging.debug("j Shape Transposed: ", j.shape)

        # Map to [(-1, -1), (1, 1)] and then NDC (scaled by focal length):
        #   x: (i - width/2) / focal
        #   y: -(j - height/2) / focal
        #   z: -1 (-1 is camera's forward)
        directions = torch.stack(
            [
                (i - width * 0.5) / focal_length,
                -(j - height * 0.5) / focal_length,
                -torch.ones_like(i),
            ],
            dim=-1,
        )

        # Convert to world coords
        # Apply camera pose to directions
        rays_d = torch.sum(directions[..., None, :] * c2w[:3, :3], dim=-1)
        logging.debug("Ray Directions:")
        logging.debug("Shape: ", rays_d.shape)
        logging.debug("Sample: ", rays_d[height // 2, width // 2, :])
        logging.debug("")

        # Origin is same for all directions (the optical center)
        rays_o = c2w[:3, -1].expand(rays_d.shape)

        logging.debug("Ray Origins:")
        logging.debug("Shape: ", rays_o.shape)
        logging.debug("Sample: ", rays_o[height // 2, width // 2, :])
        logging.debug("")

        return rays_o, rays_d


if __name__ == "__main__":
    r = Renderer()
    with torch.no_grad():
        r.get_rays(r.height, r.width, r.focal, r.testpose)
