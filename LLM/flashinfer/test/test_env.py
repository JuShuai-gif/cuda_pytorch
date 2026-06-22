#!/usr/bin/env python3
"""
针对 flashinfer.jit.env 模块的单元测试。
测试环境变量和模块检测功能。
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, Mock
import importlib

# 添加父目录到路径，以便导入 flashinfer 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEnvModule(unittest.TestCase):
    """测试 env.py 模块的功能"""

    def setUp(self):
        """在每个测试之前保存原始环境变量和模块状态"""
        self.original_env = os.environ.copy()
        # 保存原始模块引用，以便之后恢复
        if "flashinfer.jit.env" in sys.modules:
            self.original_module = sys.modules["flashinfer.jit.env"]
        else:
            self.original_module = None

        # 模拟 torch 模块以避免导入错误
        self.torch_mock = Mock()
        self.torch_mock.cuda = Mock()
        self.torch_mock.cuda.device_count = Mock(return_value=0)
        self.torch_mock.cuda.get_device_capability = Mock(return_value=(8, 0))
        self.torch_patch = patch.dict("sys.modules", {"torch": self.torch_mock})
        self.torch_patch.start()

        # 模拟 flashinfer.jit.cpp_ext.is_cuda_version_at_least 函数
        self.cpp_ext_mock = Mock()
        self.cpp_ext_mock.is_cuda_version_at_least = Mock(return_value=True)
        self.cpp_ext_patch = patch.dict(
            "sys.modules", {"flashinfer.jit.cpp_ext": self.cpp_ext_mock}
        )
        self.cpp_ext_patch.start()

    def tearDown(self):
        """在每个测试之后恢复环境变量和模块状态"""
        os.environ.clear()
        os.environ.update(self.original_env)
        # 恢复原始模块
        if self.original_module is not None:
            sys.modules["flashinfer.jit.env"] = self.original_module
        elif "flashinfer.jit.env" in sys.modules:
            del sys.modules["flashinfer.jit.env"]

        # 停止模拟
        self.torch_patch.stop()
        self.cpp_ext_patch.stop()

    def _get_env_module(self):
        """导入并返回 flashinfer.jit.env 模块，确保重新加载以获取最新环境变量"""
        import flashinfer.jit.env as env_module

        importlib.reload(env_module)
        return env_module

    def test_has_flashinfer_jit_cache_true(self):
        """测试 has_flashinfer_jit_cache 返回 True 的情况"""
        with patch("importlib.util.find_spec") as mock_find_spec:
            mock_find_spec.return_value = MagicMock()  # 模拟模块存在
            env_module = self._get_env_module()
            self.assertTrue(env_module.has_flashinfer_jit_cache())
            mock_find_spec.assert_called_with("flashinfer_jit_cache")

    def test_has_flashinfer_jit_cache_false(self):
        """测试 has_flashinfer_jit_cache 返回 False 的情况"""
        with patch("importlib.util.find_spec") as mock_find_spec:
            mock_find_spec.return_value = None  # 模拟模块不存在
            env_module = self._get_env_module()
            self.assertFalse(env_module.has_flashinfer_jit_cache())

    def test_has_flashinfer_cubin_true(self):
        """测试 has_flashinfer_cubin 返回 True 的情况"""
        with patch("importlib.util.find_spec") as mock_find_spec:
            mock_find_spec.return_value = MagicMock()  # 模拟模块存在
            env_module = self._get_env_module()
            self.assertTrue(env_module.has_flashinfer_cubin())
            mock_find_spec.assert_called_with("flashinfer_cubin")

    def test_has_flashinfer_cubin_false(self):
        """测试 has_flashinfer_cubin 返回 False 的情况"""
        with patch("importlib.util.find_spec") as mock_find_spec:
            mock_find_spec.return_value = None  # 模拟模块不存在
            env_module = self._get_env_module()
            self.assertFalse(env_module.has_flashinfer_cubin())

    def test_flashinfer_base_dir_default(self):
        """测试 FLASHINFER_BASE_DIR 默认值（用户主目录）"""
        # 确保环境变量不存在
        if "FLASHINFER_WORKSPACE_BASE" in os.environ:
            del os.environ["FLASHINFER_WORKSPACE_BASE"]
        env_module = self._get_env_module()
        import pathlib

        expected = pathlib.Path.home()
        self.assertEqual(env_module.FLASHINFER_BASE_DIR, expected)

    def test_flashinfer_base_dir_custom(self):
        """测试通过环境变量设置 FLASHINFER_BASE_DIR"""
        test_path = "/tmp/test_flashinfer_workspace"
        os.environ["FLASHINFER_WORKSPACE_BASE"] = test_path
        env_module = self._get_env_module()
        import pathlib

        expected = pathlib.Path(test_path)
        self.assertEqual(env_module.FLASHINFER_BASE_DIR, expected)

    def test_flashinfer_cache_dir(self):
        """测试 FLASHINFER_CACHE_DIR 是否正确构建"""
        test_path = "/tmp/test_flashinfer_workspace"
        os.environ["FLASHINFER_WORKSPACE_BASE"] = test_path
        env_module = self._get_env_module()
        import pathlib

        expected = pathlib.Path(test_path) / ".cache" / "flashinfer"
        self.assertEqual(env_module.FLASHINFER_CACHE_DIR, expected)

    def test_package_root(self):
        """测试 _package_root 是否正确指向包根目录"""
        env_module = self._get_env_module()
        import pathlib

        # 计算预期路径：env.py 所在目录的父目录的父目录
        expected = pathlib.Path(env_module.__file__).resolve().parents[1]
        self.assertEqual(env_module._package_root, expected)


if __name__ == "__main__":
    unittest.main()
