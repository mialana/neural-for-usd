from imports import *


class Visualizer:
    def __init__(self):
        if not os.path.exists("tiny_nerf_data.npz"):
            wget.download(
                "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz"
            )

        npz_file: NpzFile = np.load("tiny_nerf_data.npz")
        self.images: np.ndarray = npz_file.get("images", np.ones((1, 1, 1, 1)))
        self.poses: np.ndarray = npz_file.get("poses", np.ones((1, 1, 1)))
        self.focal = npz_file.get("focal", -1.0)

        self.height, self.width = self.images.shape[1:3]
        self.near, self.far = 2.0, 6.0

        self.n_training = 100
        self.testimg_idx = 101
        self.testimg, self.testpose = (
            self.images[self.testimg_idx],
            self.poses[self.testimg_idx],
        )

        plt.imshow(self.testimg)
        plt.show()

        logging.debug("Pose:")
        logging.debug(self.testpose)

        self.pts = None
        self.rays_o = None
        self.rays_d = None

    def visualizeRays(self):
        # pose[:3, :3] is each pose's transformation matrix
        # multiplies [0, 0, -1] (camera's view vector in camera space) by x-form
        # effectively gets camera's world-space view vector
        dirs = np.stack(
            [np.sum([0, 0, -1] * pose[:3, :3], axis=-1) for pose in self.poses]
        )

        origins = self.poses[:, :3, -1]

        logging.debug(f"Dirs shape: {dirs.shape}")  # exp: (106, 3)
        logging.debug(f"Origins shape: {origins.shape}")  # exp: (106, 3)

        fg = plt.figure(figsize=(12, 8))
        ax: Axes3D = fg.add_subplot(projection="3d")

        # ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)
        # where x,y,z is array of arrow origins, respectively,
        # and u, v, w are of arrow directions
        _ = ax.quiver(
            origins[..., 0].flatten(),  # selects the x-coordinate of all 106 origins
            origins[..., 1].flatten(),
            origins[..., 2].flatten(),
            dirs[..., 0].flatten(),
            dirs[..., 1].flatten(),
            dirs[..., 2].flatten(),
            length=0.5,
            normalize=True,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("z")
        plt.show()

    def get_rays(
        height: int, width: int, focal_length: float, c2w: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Find origin and direction of rays through every pixel and camera origin.
        """

        # Apply pinhole camera model to gather directions at each pixel
        i, j = torch.meshgrid(
            torch.arange(width, dtype=torch.float32).to(c2w),
            torch.arange(height, dtype=torch.float32).to(c2w),
            indexing="ij",
        )
        i, j = i.transpose(-1, -2), j.transpose(-1, -2)
        directions = torch.stack(
            [
                (i - width * 0.5) / focal_length,
                -(j - height * 0.5) / focal_length,
                -torch.ones_like(i),
            ],
            dim=-1,
        )

        # Apply camera pose to directions
        rays_d = torch.sum(directions[..., None, :] * c2w[:3, :3], dim=-1)

        # Origin is same for all directions (the optical center)
        rays_o = c2w[:3, -1].expand(rays_d.shape)
        return rays_o, rays_d


if __name__ == "__main__":
    v = Visualizer()
    v.visualizeRays()
