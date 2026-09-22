# ============================================================
# tests/test_tools.py —— 工具层单测
# ============================================================

import pytest

from agent.tools import validate_args, query_city_info, get_user_city, get_weather


class TestValidateArgs:
    def test_unknown_tool(self):
        ok, err = validate_args("not_a_tool", {})
        assert not ok
        assert "未知工具" in err

    def test_missing_required(self):
        ok, err = validate_args("get_weather", {})
        assert not ok
        assert "city" in err

    def test_wrong_type(self):
        ok, err = validate_args("get_weather", {"city": 123})
        assert not ok
        assert "类型错误" in err

    def test_ok(self):
        ok, err = validate_args("get_weather", {"city": "北京"})
        assert ok
        assert err == ""


class TestTools:
    def test_query_city_found(self, tmp_db):
        result = query_city_info("北京")
        assert "北京市" in result
        assert "2189" in result

    def test_query_city_not_found(self, tmp_db):
        result = query_city_info("不存在的城市")
        assert "没有找到" in result

    def test_get_user_city(self, tmp_db):
        result = get_user_city("1001")
        assert "张三" in result
        assert "北京" in result

    def test_weather_invalid_city(self):
        result = get_weather("火星")
        assert "不是地球城市" in result