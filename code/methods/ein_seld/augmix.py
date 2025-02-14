# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Reference implementation of AugMix's data augmentation method in numpy."""
import methods.ein_seld.augmentations as augmentations
import numpy as np
import torch
from methods.ein_seld.rotate import Rotation
import copy

def apply_op(op, batch_x,batch_target):

  batch_x2,batch_target2= op(batch_x,batch_target)
  return batch_x2,batch_target2


def augment_and_mix(batch_x,batch_target, width=3, depth=-1):
  """Perform AugMix augmentations and compute mixture.

  Args:
    width: Width of augmentation chain
    depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
      from [1, 3]
"""

  batch_x_ground=batch_x.clone()
  batch_target_ground=copy.deepcopy(batch_target)
  for i in range(width):
    batch_x_aug = batch_x.clone()
    batch_target_aug = copy.deepcopy(batch_target)
    d = depth if depth > 0 else np.random.randint(1, 4)
    for _ in range(d):
      op = np.random.choice(augmentations.augmentations)
      batch_x_aug,batch_target_aug = apply_op(op,batch_x_aug,batch_target_aug)
    # Preprocessing commutes since all coefficients are convex
    batch_x_ground=torch.cat((batch_x_ground, batch_x_aug), dim=0)
    for kv in batch_target_ground:
        if(type(batch_target_ground[kv])==list):
            batch_target_ground[kv].append(batch_target_aug[kv])

        else:
            batch_target_ground[kv]=torch.cat((batch_target_ground[kv], batch_target_aug[kv]), dim=0)
  

  return batch_x_ground,batch_target_ground
