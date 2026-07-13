# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any

import pytest

import vllm_ascend.patch.hunyuan_vl_processor_compat as compat


def test_v023_translates_bundled_image_processor_registration(monkeypatch):
    from transformers import AutoImageProcessor
    from vllm.transformers_utils.configs.hunyuan_vl import HunYuanVLConfig

    class HunYuanVLImageProcessor:
        pass

    hunyuan_vision = object()
    registrations: list[tuple[Any, Any, bool]] = []

    def register(
        config_class: Any,
        slow_image_processor_class: Any = None,
        *,
        exist_ok: bool = False,
    ) -> None:
        registrations.append(
            (
                config_class,
                slow_image_processor_class,
                exist_ok,
            )
        )

    def import_hunyuan_vision(name: str) -> object:
        assert name == "vllm.model_executor.models.hunyuan_vision"
        AutoImageProcessor.register(
            "HunYuanVLImageProcessor",
            HunYuanVLImageProcessor,
        )
        return hunyuan_vision

    monkeypatch.setattr(AutoImageProcessor, "register", staticmethod(register))
    monkeypatch.setattr(compat.importlib, "import_module", import_hunyuan_vision)

    assert compat._import_v023_hunyuan_vision() is hunyuan_vision
    assert AutoImageProcessor.register is register
    assert registrations == [
        (
            HunYuanVLConfig,
            HunYuanVLImageProcessor,
            True,
        )
    ]


def test_v023_restores_register_after_unexpected_registration(monkeypatch):
    from transformers import AutoImageProcessor

    class UnexpectedProcessor:
        pass

    def register(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("The unexpected registration must not reach Transformers")

    def import_hunyuan_vision(_name: str) -> object:
        AutoImageProcessor.register("UnexpectedProcessor", UnexpectedProcessor)
        return object()

    monkeypatch.setattr(AutoImageProcessor, "register", staticmethod(register))
    monkeypatch.setattr(compat.importlib, "import_module", import_hunyuan_vision)

    with pytest.raises(RuntimeError, match="Unexpected v0.23 Hunyuan image-processor registration"):
        compat._import_v023_hunyuan_vision()

    assert AutoImageProcessor.register is register


def test_installer_keeps_v023_bundled_processor_protocol(monkeypatch):
    def bundled_call_protocol(*_args: Any, **_kwargs: Any) -> str:
        return "bundled"

    class BundledProcessor:
        pass

    class FakeProcessingInfo:
        pass

    class FakeMultiModalProcessor:
        _call_hf_processor = bundled_call_protocol

    hunyuan_vision = SimpleNamespace(
        HunYuanVLProcessor=BundledProcessor,
        HunYuanVLProcessingInfo=FakeProcessingInfo,
        HunYuanVLMultiModalProcessor=FakeMultiModalProcessor,
    )

    def fail_main_path(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("The v0.23 path must not install main's native processor patches")

    monkeypatch.setattr(compat, "vllm_version_is", lambda version: version == "0.23.0")
    monkeypatch.setattr(
        compat,
        "_import_v023_hunyuan_vision",
        lambda: hunyuan_vision,
    )
    monkeypatch.setattr(compat, "_remove_stale_registry_entries", fail_main_path)

    compat.install_hunyuan_vl_processor_compat()

    processor_args: list[tuple[Any, dict[str, Any]]] = []

    def get_processor(processor_class: Any, **kwargs: Any) -> object:
        processor_args.append((processor_class, kwargs))
        return object()

    processing_info = SimpleNamespace(
        ctx=SimpleNamespace(get_hf_processor=get_processor),
    )
    get_hf_processor = vars(FakeProcessingInfo)["get_hf_processor"]
    get_hf_processor(processing_info, use_fast=True, min_pixels=128)

    assert processor_args == [
        (
            BundledProcessor,
            {"min_pixels": 128, "backend": "pil"},
        )
    ]
    assert vars(FakeMultiModalProcessor)["_call_hf_processor"] is bundled_call_protocol


def test_installer_cleans_main_registry_before_model_patch(monkeypatch):
    import vllm.model_executor.models as vllm_models

    def native_get_prompt_updates(*_args: Any, **_kwargs: Any) -> str:
        return "native"

    class FakeMultiModalProcessor:
        _get_prompt_updates = native_get_prompt_updates

    hunyuan_vision = SimpleNamespace(
        HunYuanVLMultiModalProcessor=FakeMultiModalProcessor,
    )
    calls: list[Any] = []

    def clean_registry() -> bool:
        calls.append("registry")
        return True

    def patch_loader(module: Any, processor_class: Any) -> None:
        calls.append(("loader", module, processor_class))

    monkeypatch.setattr(compat, "vllm_version_is", lambda _version: False)
    monkeypatch.setattr(
        compat,
        "_remove_stale_registry_entries",
        clean_registry,
    )
    monkeypatch.setattr(vllm_models, "hunyuan_vision", hunyuan_vision, raising=False)
    monkeypatch.setattr(
        compat,
        "_patch_hunyuan_processor_loader",
        patch_loader,
    )

    compat.install_hunyuan_vl_processor_compat()

    assert calls == [
        "registry",
        (
            "loader",
            hunyuan_vision,
            compat._HunYuanVLProcessorCompat,
        ),
    ]
    assert FakeMultiModalProcessor._get_prompt_updates is native_get_prompt_updates


def test_registers_hunyuan_tokenizer_schema_without_changing_ids():
    class FakeTokenizer:
        pad_token = compat._HUNYUAN_VL_SPECIAL_TOKENS["pad_token"]
        pad_token_id = compat._HUNYUAN_VL_SPECIAL_TOKEN_IDS["pad_token"]

        def __init__(self) -> None:
            self.registrations: list[dict[str, str]] = []

        def _set_model_specific_special_tokens(self, special_tokens: dict[str, str]) -> None:
            self.registrations.append(special_tokens)
            for name, token in special_tokens.items():
                setattr(self, name, token)
                setattr(self, f"{name}_id", compat._HUNYUAN_VL_SPECIAL_TOKEN_IDS[name])

    tokenizer = FakeTokenizer()

    compat._register_hunyuan_tokenizer_special_tokens(tokenizer)
    compat._register_hunyuan_tokenizer_special_tokens(tokenizer)

    assert tokenizer.registrations == [compat._HUNYUAN_VL_EXTRA_SPECIAL_TOKENS]


def test_rejects_hunyuan_token_id_mismatch():
    tokenizer = SimpleNamespace(
        **compat._HUNYUAN_VL_SPECIAL_TOKENS,
        **{f"{name}_id": token_id for name, token_id in compat._HUNYUAN_VL_SPECIAL_TOKEN_IDS.items()},
    )
    tokenizer.image_token_id = 1

    with pytest.raises(ValueError, match="does not match the model vocabulary"):
        compat._register_hunyuan_tokenizer_special_tokens(tokenizer)


def test_compat_processor_registers_schema_before_native_init(monkeypatch):
    tokenizer = object()
    calls: list[tuple[Any, ...]] = []

    monkeypatch.setattr(
        compat,
        "_register_hunyuan_tokenizer_special_tokens",
        lambda value: calls.append(("register", value)),
    )

    def native_init(
        self: Any,
        image_processor: Any = None,
        tokenizer: Any = None,
        chat_template: Any = None,
        cat_extra_token: bool = True,
        **kwargs: Any,
    ) -> None:
        calls.append(("native", tokenizer, cat_extra_token, kwargs))

    monkeypatch.setattr(compat.HunYuanVLProcessor, "__init__", native_init)

    compat._HunYuanVLProcessorCompat(
        image_processor=object(),
        tokenizer=tokenizer,
        cat_extra_token=False,
        custom=True,
    )

    assert calls == [
        ("register", tokenizer),
        ("native", tokenizer, False, {"custom": True}),
    ]


def test_main_removes_only_stale_registry_entries(monkeypatch):
    import vllm.transformers_utils.processors as vllm_processors

    registry = {
        **compat._STALE_PROCESSOR_MODULES,
        "OtherProcessor": "vllm.transformers_utils.processors.other",
    }
    exported_names = [*registry]
    monkeypatch.setattr(vllm_processors, "_CLASS_TO_MODULE", registry)
    monkeypatch.setattr(vllm_processors, "__all__", exported_names)

    assert compat._remove_stale_registry_entries()
    assert registry == {
        "OtherProcessor": "vllm.transformers_utils.processors.other",
    }
    assert exported_names == ["OtherProcessor"]
    assert not compat._remove_stale_registry_entries()


def test_main_rejects_unexpected_registry_replacement(monkeypatch):
    import vllm.transformers_utils.processors as vllm_processors

    registry = {
        "HunYuanVLProcessor": "future.hunyuan_vl",
    }
    monkeypatch.setattr(vllm_processors, "_CLASS_TO_MODULE", registry)
    monkeypatch.setattr(vllm_processors, "__all__", ["HunYuanVLProcessor"])

    with pytest.raises(RuntimeError, match="Unexpected vLLM processor registry entry"):
        compat._remove_stale_registry_entries()

    assert registry == {"HunYuanVLProcessor": "future.hunyuan_vl"}
