# SPDX-License-Identifier: Apache-2.0

from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.33.0"):
    from ._release_prefix_cache_cp_patches_cases import *  # noqa: F403
else:
    from ._native_prefix_cache_cp_patches_cases import *  # noqa: F403
