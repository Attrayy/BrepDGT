import numpy as np
import torch


def apply_affine_transform(points, tx):
    if type(points) == np.ndarray:
        if len(points.shape) == 1:
            points = points[np.newaxis, :]
        points = np.concatenate([points, np.ones_like(points[:, :1])], -1)
        return np.ascontiguousarray((tx @ points.T).T[:, :3])
    elif type(points) == torch.Tensor:
        points = torch.cat([points, torch.ones_like(points[:, :1])], -1)
        return ((tx @ points.T).T[:, :3]).contiguous()


def affine_transform_matrix(rotation, translation, scale):
    ret = np.eye(4)
    if rotation is None:
        rotation = np.eye(3)
    if scale is None:
        scale = np.ones(3)

    rs = rotation @ np.diag(np.squeeze(scale))
    ret[:3, :3] = rs
    if translation is not None:
        ret[:3, -1] = translation
    return ret


class Rot90Flip:
    """
    Rotate 90 degrees for each axis and flip
    """

    def __init__(self):
        self.aa_rot_txs = self._get_unique_aa_rots()  # 24 x 4 x 4 matrices

    def _get_unique_aa_rots(self):
        """
        Returns a list of 24 unique axis-aligned rotation matrices (symmetry group of the cube).
        """
        # Identity matrix (0° rotation)
        I = np.eye(3)

        # 90°, 180°, 270° rotations about each axis
        x_rots = [
            I,
            np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
            np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
            np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
        ]
        y_rots = [
            I,
            np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
            np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
            np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
        ]
        z_rots = [
            I,
            np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
            np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
            np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
        ]
        rotations = []
        for x_rot in x_rots:
            for y_rot in y_rots:
                for z_rot in z_rots:
                    rot = x_rot @ y_rot @ z_rot
                    rotations.append(rot)

        rotations = np.stack(rotations, axis=0)

        # there should only be 24 unique axis aligned rotations in 3d
        unique_rotations = np.unique(rotations, axis=0)
        assert len(unique_rotations) == 24, f"got {len(unique_rotations)} rotations"

        # make it 4x4 affine transform matrix
        txs = np.stack([np.eye(4) for _ in range(len(unique_rotations))])
        txs[:, :3, :3] = unique_rotations

        return txs

    def get_rand_tx(self, enable_rotation=True, enable_flip=True):
        """
        Get random transformation matrix with configurable rotation and flip

        Args:
            enable_rotation (bool): Whether to apply random 90-degree rotations
            enable_flip (bool): Whether to apply random flips
        Returns:
            4x4 transformation matrix
        """
        if enable_rotation:
            rot_idx = np.random.randint(0, len(self.aa_rot_txs))
            rot_tx = self.aa_rot_txs[rot_idx]
        else:
            rot_tx = np.eye(4)

        if enable_flip:
            flips = np.random.choice([1, -1], size=(4,))
            flips[3] = 1.0
            flip_tx = np.diag(flips)
        else:
            flip_tx = np.eye(4)

        tx = rot_tx @ flip_tx
        return tx


class ScaleTranslate:
    """
    Randomly scale and translate 3D models within configured ranges
    """

    def __init__(
        self, scale_min=0.5, scale_max=1.2, translate_min=-0.2, translate_max=0.2
    ):
        """
        Args:
            scale_range (tuple): Min and max scale factors (default: 0.5 to 1.2)
            translate_range (tuple): Min and max translation in each axis (default: -0.2 to 0.2)
        """
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.translate_min = translate_min
        self.translate_max = translate_max

    def get_rand_tx(self):
        # Random uniform scale for all axes
        scale = np.random.uniform(self.scale_min, self.scale_max)
        scale_matrix = np.eye(4) * scale
        scale_matrix[3, 3] = 1.0

        # Random translation for each axis
        translation = np.random.uniform(self.translate_min, self.translate_max, size=3)

        # Create translation matrix
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = translation

        # Combine scale and translation
        tx = translation_matrix @ scale_matrix
        return tx
