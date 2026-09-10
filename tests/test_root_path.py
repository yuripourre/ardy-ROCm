# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from ardy.root_path import interpolate_root_path


def test_interpolate_root_path_linear():
    frame_indices = np.array([0, 10], dtype=np.float64)
    root_positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 20.0]], dtype=np.float64)
    t = np.array([0, 5, 10])
    path = interpolate_root_path(frame_indices, root_positions, t, smooth=False)
    assert path.shape == (3, 3)
    np.testing.assert_allclose(path[5, 0], 5.0)
    np.testing.assert_allclose(path[5, 2], 10.0)
    np.testing.assert_allclose(path[:, 1], 0.0)
