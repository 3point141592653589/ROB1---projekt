from ctu_crs import CRS97
import numpy as np


class CRS97Patch(CRS97):
    def __init__(
        self,
        tty_dev: str | None = "/dev/mars",
        baudrate: int = 19200,
        dh_offset=None,
        **crs_kwargs,
    ):
        super().__init__(tty_dev, baudrate, **crs_kwargs)
        self._original_dh_offset = self.dh_offset
        if dh_offset is not None:
            self.dh_offset = dh_offset

    def _ik_flange_pos(
        self, flange_pos: np.ndarray, singularity_theta1=0
    ) -> list[np.ndarray]:
        return [
            q - (self.dh_offset - self._original_dh_offset)[:3]
            for q in super()._ik_flange_pos(flange_pos, singularity_theta1)
        ]

    def ik(self, pose: np.ndarray) -> list[np.ndarray]:
        ret = []
        for q in super().ik(pose):
            q[3:] -= (self.dh_offset - self._original_dh_offset)[3:]
            ret.append(q)
        return ret
