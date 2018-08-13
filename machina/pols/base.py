# Copyright 2018 DeepX Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import torch
import torch.nn as nn

class BasePol(nn.Module):
    def __init__(self, ob_space, ac_space, normalize_ac=True):
        nn.Module.__init__(self)
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.normalize_ac = normalize_ac

    def convert_ac_for_real(self, x):
        lb, ub = self.ac_space.low, self.ac_space.high
        if self.normalize_ac:
            x = lb + (x + 1.) * 0.5 * (ub - lb)
            x = np.clip(x, lb, ub)
        else:
            x = np.clip(x, lb, ub)
        return x

    def reset(self):
        pass
